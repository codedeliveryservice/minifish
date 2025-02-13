use bullet_lib::{
    inputs, optimiser, outputs, Activation, Loss, TrainerBuilder,
};

const HIDDEN_SIZE: usize = 64;
const QA: i16 = 101;
const QB: i16 = 160;

fn main() {
    let quantized_output: String = format!("/data/HL{}-qa{}-qb{}-S2-T77novT79maraprmay.nnue", HIDDEN_SIZE, QA, QB);
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMPE(2.6))
        .input(inputs::ChessBucketsMirrored::new([0; 32]))
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    trainer.load_from_checkpoint("./checkpoints/HL64-hm--S2-T77nov-T79-wdl-pdist-see-ge0-lr125--S1-UHO-no-wm-lr15-pdist-re2-120/");
    let _ = trainer.save_quantised(&quantized_output);
    println!("{}", &quantized_output);
}
