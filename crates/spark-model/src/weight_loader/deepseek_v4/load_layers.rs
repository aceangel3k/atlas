// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_map::{dense, dense_auto, quantize_to_nvfp4};

pub fn load_all_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let n = config.num_hidden_layers;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let mut layers = Vec::with_capacity(n);
    let mut yarn_inv_freq = DevicePtr::NULL;

    for i in 0..n {
        let lp = config.layer_prefix(i);
        let ap = format!("{lp}.self_attn");

        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

        let wq_a = dense_auto(store, &format!("{ap}.q_a_proj.weight"), gpu)?;
        let wq_a_nvfp4 = Some(quantize_to_nvfp4(
            &wq_a,
            config.q_lora_rank,
            config.hidden_size,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?);
        let wq_b = dense_auto(store, &format!("{ap}.q_b_proj.weight"), gpu)?;
        let wq_b_nvfp4 = Some(quantize_to_nvfp4(
            &wq_b,
            config.num_attention_heads * config.head_dim,
            config.q_lora_rank,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?);
        let q_a_norm = dense(store, &format!("{ap}.q_a_layernorm.weight"))?;

        let wkv_a = dense_auto(store, &format!("{ap}.kv_a_proj_with_mqa.weight"), gpu)?;
        let wkv_a_nvfp4 = Some(quantize_to_nvfp4(
            &wkv_a,
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?);
        let wkv_b = dense_auto(store, &format!("{ap}.kv_b_proj.weight"), gpu)?;
        let kv_a_norm = dense(store, &format!("{ap}.kv_a_layernorm.weight"))?;

        let o_dense = dense_auto(store, &format!("{ap}.o_proj.weight"), gpu)?;
        let o_nvfp4 = Some(quantize_to_nvfp4(
            &o_dense,
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?);

        let (w_uk_t, w_uv, wq_b_rope, w_uk_host) =
            super::compute::build_per_head_views(&wkv_b, &wq_b, config, gpu)?;
        let w_qk_absorbed = super::compute::build_w_qk_absorbed(&wq_b, &w_uk_t, config, gpu)?;
        let (w_uk_block_diag, w_uv_block_diag) =
            super::compute::build_block_diagonals(&w_uk_host, &w_uv, config, gpu)?;
        yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

        let layer = super::assemble::assemble_layer(
            i,
            input_norm,
            post_attn_norm,
            wq_a,
            wq_a_nvfp4,
            wq_b,
            wq_b_nvfp4,
            q_a_norm,
            wkv_a,
            wkv_a_nvfp4,
            wkv_b,
            kv_a_norm,
            o_dense,
            o_nvfp4,
            w_uk_t,
            w_uv,
            wq_b_rope,
            w_qk_absorbed,
            w_uk_block_diag,
            w_uv_block_diag,
            yarn_inv_freq,
            store,
            config,
            gpu,
            layer_kv_dtypes,
        )?;
        layers.push(layer);
    }
    Ok(layers)
}
