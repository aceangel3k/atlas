// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::MoeLayer;
use crate::layers::qwen3_attention::{MlaWeights, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, QuantizeCtx, QuantizedWeight,
    detect_nvfp4_variant, load_moe, quantize_to_nvfp4,
};

#[allow(clippy::too_many_arguments)]
pub fn assemble_layer(
    layer_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    wq_a: DenseWeight,
    wq_a_nvfp4: Option<QuantizedWeight>,
    wq_b: DenseWeight,
    wq_b_nvfp4: Option<QuantizedWeight>,
    q_a_norm: DenseWeight,
    wkv_a: DenseWeight,
    wkv_a_nvfp4: Option<QuantizedWeight>,
    wkv_b: DenseWeight,
    kv_a_norm: DenseWeight,
    o_dense: DenseWeight,
    o_nvfp4: Option<QuantizedWeight>,
    w_uk_t: DenseWeight,
    w_uv: DenseWeight,
    wq_b_rope: DenseWeight,
    w_qk_absorbed: DenseWeight,
    w_uk_block_diag: DenseWeight,
    w_uv_block_diag: DenseWeight,
    yarn_inv_freq: DevicePtr,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Box<dyn TransformerLayer>> {
    let lp = config.layer_prefix(layer_idx);
    let h = config.hidden_size;
    let kv_dtype = layer_kv_dtypes
        .get(layer_idx)
        .copied()
        .unwrap_or(KvCacheDtype::Bf16);

    // ── MoE FFN ──
    let variant = detect_nvfp4_variant(store, config);
    let qctx = QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };
    let moe_weights = load_moe(store, &lp, config.num_experts, gpu, config, variant, qctx)?;
    let gate_nvfp4 = Some(quantize_to_nvfp4(
        &moe_weights.gate,
        config.num_experts,
        config.hidden_size,
        gpu,
        qctx.absmax_k,
        qctx.quantize_k,
        qctx.stream,
    )?);
    let moe = MoeLayer::new(moe_weights, config.num_experts, gate_nvfp4, gpu, config)?;

    // ── MLA weights ──
    let mla = MlaWeights {
        wq_a,
        wq_a_nvfp4,
        wq_b,
        wq_b_nvfp4,
        q_a_norm,
        wkv_a,
        wkv_a_nvfp4,
        wkv_b,
        kv_a_norm,
        wkv_a_rope: DenseWeight {
            weight: wkv_a.weight.offset(config.kv_lora_rank * h * 2),
        },
        wkv_a_merged: DenseWeight {
            weight: wkv_a.weight,
        },
        wo: o_dense,
        wo_nvfp4: o_nvfp4,
        w_uk_t,
        w_uv,
        wq_b_rope,
        w_qk_absorbed,
        w_uk_block_diag,
        w_uv_block_diag,
        yarn_inv_freq,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        nope: config.qk_nope_head_dim,
        rope: config.qk_rope_head_dim,
        v_dim: config.v_head_dim,
    };

    // ── Attention dummy + layer ──
    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        o_proj: QuantizedWeight::null(),
        q_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        FfnComponent::Moe(moe),
        layer_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        0,
        config,
    )?;
    layer.set_mla_weights(mla);
    Ok(Box::new(layer))
}
