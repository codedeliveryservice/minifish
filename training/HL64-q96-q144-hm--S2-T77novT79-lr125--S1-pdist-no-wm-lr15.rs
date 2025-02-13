use bullet_lib::{
    inputs, loader, lr, optimiser, outputs, wdl, Activation, LocalSettings, Loss,
    TrainerBuilder, TrainingSchedule, TrainingSteps
};

const HIDDEN_SIZE: usize = 64;
const QA: i16 = 96;
const QB: i16 = 144;

const LOSS_POW: f32 = 2.6;
const EVAL_SCALE: f32 = 340.0;

const S1_DATASET_PATHS: &[&str] = &[
    "/data/UHO.pdist.iter-1.bullet.bin",
    "/data/UHO.pdist.iter-2.bullet.bin",
    "/data/UHO.pdist.iter-3.bullet.bin",
    "/data/UHO.pdist.iter-4.bullet.bin",
    "/data/UHO.pdist.iter-5.bullet.bin",
    "/data/UHO.pdist.iter-6.bullet.bin",
    "/data/UHO.pdist.iter-7.bullet.bin",
    "/data/UHO.pdist.iter-8.bullet.bin",
    "/data/UHO.pdist.iter-9.bullet.bin",
    "/data/UHO.pdist.iter-10.bullet.bin",
];

const S2_DATASET_PATHS: &[&str] = &[
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-1.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-2.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-3.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-4.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-5.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-6.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-7.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-8.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-9.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-10.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-11.bullet.bin",
    "/data/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-12.bullet.bin",
];

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMPE(LOSS_POW))
        .input(inputs::ChessBucketsMirrored::new([0; 32]))
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let optimiser_params =
        optimiser::AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };

    trainer.set_optimiser_params(optimiser_params);

    // S1
    const S1_SB: usize = 100;
    let s1_net_id = format!(
        "HL{}-hm--S1-UHO-no-wm-lr15-pdist-re2",
        HIDDEN_SIZE).to_string();

    let schedule = TrainingSchedule {
        net_id: s1_net_id.clone(),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 6_104, start_superbatch: 1, end_superbatch: S1_SB },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::LinearDecayLR { initial_lr: 0.0015, final_lr: 0.0, final_superbatch: S1_SB },
        save_rate: 10,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(S1_DATASET_PATHS);
    trainer.run(&schedule, &settings, &data_loader);


    // S2
    const S2_SB: usize = 120;
    let s2_net_id = format!(
        "HL{}-hm--S2-T77nov-T79-wdl-pdist-see-ge0-lr125--S1-UHO-no-wm-lr15-pdist-re2",
        HIDDEN_SIZE).to_string();

    let schedule = TrainingSchedule {
        net_id: s2_net_id.clone(),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 6_104, start_superbatch: 1, end_superbatch: S2_SB },
        wdl_scheduler: wdl::LinearWDL { start: 0.0, end: 0.1 },
        lr_scheduler: lr::LinearDecayLR { initial_lr: 0.000125, final_lr: 0.0, final_superbatch: S2_SB },
        save_rate: 10,
    };

    trainer.load_from_checkpoint(&format!("./checkpoints/{}-{}", &s1_net_id, S1_SB));

    let data_loader = loader::DirectSequentialDataLoader::new(S2_DATASET_PATHS);

    trainer.run(&schedule, &settings, &data_loader);
}
