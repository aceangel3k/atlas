// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 weight loader (MLA + MoE).
//!
//! Implements full layer loading for DeepSeek-V4-Flash, reusing the same
//! MLA attention pattern as Mistral Small 4 with DeepSeek weight naming.

mod assemble;
mod compute;
mod load_layers;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

pub struct DeepSeekV4WeightLoader;

impl ModelWeightLoader for DeepSeekV4WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_all_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else if config.tie_word_embeddings {
            self.load_embedding(store, config)
        } else {
            anyhow::bail!("DeepSeek-V4: lm_head not found")
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
