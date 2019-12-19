class MSVDSplitConfig:
    model = "MSVD_InceptionV4"

    video_fpath = "data/MSVD/features/{}.hdf5".format(model)
    caption_fpath = "data/MSVD/metadata/MSR Video Description Corpus.csv"

    train_video_fpath = "data/MSVD/features/{}_train.hdf5".format(model)
    val_video_fpath = "data/MSVD/features/{}_val.hdf5".format(model)
    test_video_fpath = "data/MSVD/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "data/MSVD/metadata/train.csv"
    val_metadata_fpath = "data/MSVD/metadata/val.csv"
    test_metadata_fpath = "data/MSVD/metadata/test.csv"


class MSRVTTSplitConfig:
    model = "MSR-VTT_InceptionV4"

    video_fpath = "data/MSR-VTT/features/{}.hdf5".format(model)
    train_val_caption_fpath = "data/MSR-VTT/metadata/train_val_videodatainfo.json"
    test_caption_fpath = "data/MSR-VTT/metadata/test_videodatainfo.json"

    train_video_fpath = "data/MSR-VTT/features/{}_train.hdf5".format(model)
    val_video_fpath = "data/MSR-VTT/features/{}_val.hdf5".format(model)
    test_video_fpath = "data/MSR-VTT/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "data/MSR-VTT/metadata/train.json"
    val_metadata_fpath = "data/MSR-VTT/metadata/val.json"
    test_metadata_fpath = "data/MSR-VTT/metadata/test.json"

