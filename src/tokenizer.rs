use std::path::Path;

use tokenizers::Tokenizer;

use crate::{MoeError, Result};

/// Validated tokenizer paired with a model vocabulary.
#[derive(Clone)]
pub struct MoeTokenizer {
    inner: Tokenizer,
    model_vocab_size: usize,
}

/// Owned incremental decoder that emits only stable UTF-8 chunks.
pub struct MoeDecodeStream {
    tokenizer: Tokenizer,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    skip_special_tokens: bool,
    model_vocab_size: usize,
}

impl std::fmt::Debug for MoeTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MoeTokenizer")
            .field("vocab_size", &self.inner.get_vocab_size(true))
            .field("model_vocab_size", &self.model_vocab_size)
            .finish_non_exhaustive()
    }
}

impl MoeTokenizer {
    pub fn from_file(path: impl AsRef<Path>, model_vocab_size: usize) -> Result<Self> {
        if model_vocab_size == 0 {
            return Err(MoeError::InvalidConfig(
                "model vocabulary size must be non-zero".to_string(),
            ));
        }
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|error| MoeError::Tokenizer(error.to_string()))?;
        let tokenizer_vocab_size = tokenizer.get_vocab_size(true);
        if tokenizer_vocab_size == 0 || tokenizer_vocab_size > model_vocab_size {
            return Err(MoeError::InvalidConfig(format!(
                "tokenizer vocabulary size {tokenizer_vocab_size} is outside model vocabulary size {model_vocab_size}"
            )));
        }
        Ok(Self {
            inner: tokenizer,
            model_vocab_size,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|error| MoeError::Tokenizer(error.to_string()))?;
        let token_ids = encoding.get_ids().to_vec();
        if token_ids
            .iter()
            .any(|token| *token as usize >= self.model_vocab_size)
        {
            return Err(MoeError::Tokenizer(
                "tokenizer emitted an ID outside the model vocabulary".to_string(),
            ));
        }
        Ok(token_ids)
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if token_ids
            .iter()
            .any(|token| *token as usize >= self.model_vocab_size)
        {
            return Err(MoeError::Tokenizer(
                "cannot decode an ID outside the model vocabulary".to_string(),
            ));
        }
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| MoeError::Tokenizer(error.to_string()))
    }

    pub fn decode_stream(&self, skip_special_tokens: bool) -> MoeDecodeStream {
        MoeDecodeStream {
            tokenizer: self.inner.clone(),
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            skip_special_tokens,
            model_vocab_size: self.model_vocab_size,
        }
    }
}

impl MoeDecodeStream {
    pub fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        if token_id as usize >= self.model_vocab_size {
            return Err(MoeError::Tokenizer(
                "cannot stream an ID outside the model vocabulary".to_string(),
            ));
        }
        tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer,
            vec![token_id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| MoeError::Tokenizer(error.to_string()))
    }
}

impl std::fmt::Debug for MoeDecodeStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MoeDecodeStream")
            .field("buffered_token_ids", &self.ids.len())
            .field("skip_special_tokens", &self.skip_special_tokens)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    fn write_tokenizer(path: &Path) {
        let vocab_path = path.with_file_name("vocab.json");
        std::fs::write(&vocab_path, r#"{"[UNK]":0,"hello":1,"world":2}"#).unwrap();
        let model = WordLevel::builder()
            .files(vocab_path.to_string_lossy().into_owned())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.save(path, false).unwrap();
    }

    #[test]
    fn validates_vocab_and_roundtrips_text() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        write_tokenizer(&path);

        assert!(MoeTokenizer::from_file(&path, 2).is_err());
        let tokenizer = MoeTokenizer::from_file(&path, 4).unwrap();
        let tokens = tokenizer.encode("hello world", false).unwrap();
        assert_eq!(tokens, [1, 2]);
        assert_eq!(tokenizer.decode(&tokens, true).unwrap(), "hello world");
        assert!(tokenizer.decode(&[4], true).is_err());
        let mut stream = tokenizer.decode_stream(true);
        assert_eq!(stream.step(1).unwrap().as_deref(), Some("hello"));
        assert_eq!(stream.step(2).unwrap().as_deref(), Some(" world"));
        assert!(stream.step(4).is_err());
    }

    #[test]
    fn tokenizer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MoeTokenizer>();
        assert_send_sync::<MoeDecodeStream>();
    }
}
