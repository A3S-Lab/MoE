use a3s_moe::olmoe::OlmoeTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let tokenizer_path = arguments
        .next()
        .ok_or("usage: tokenizer <tokenizer.json> <model-vocab-size> [text]")?;
    let model_vocab_size = arguments
        .next()
        .ok_or("missing model vocabulary size")?
        .parse::<usize>()?;
    let text = arguments.next().unwrap_or_else(|| "Bitcoin is".to_string());
    let tokenizer = OlmoeTokenizer::from_file(tokenizer_path, model_vocab_size)?;
    let token_ids = tokenizer.encode(&text, true)?;
    let decoded = tokenizer.decode(&token_ids, true)?;
    println!("vocab_size={}", tokenizer.vocab_size());
    println!("token_ids={token_ids:?}");
    println!("decoded={decoded}");
    Ok(())
}
