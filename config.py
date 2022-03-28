class Constant:
    SUPPLY_DEMAND_ELECTRICITY_FILES = ['./data/台灣電力公司_過去電力供需資訊_1.csv', './data/台灣電力公司_過去電力供需資訊_2.csv']
    RESERVE_MARGIN = ['./data/每日尖峰備轉容量率_1.csv', './data/每日尖峰備轉容量率_2.csv']
    OUTPUT_FILE = './submission.csv'
    TRAIN_SIZE = 0.8
    # Format
    #
    # date, operating_reserve(MW)
    # 20210323, 2557
    # 20210324, 1899
    # 20210325, 1891
    # 20210326, 1811
    # 20210327, 1903
    # 20210328, 2333
    # 20210329, 1800
    LOG_INTERVAL = 10


class Hyperparameter:
    LEARNING_RATE = 0.0002
    BATCH_SIZE = 16
    EPOCH = 1000
    PATIENCE = 5

    INPUT_SEQ_LEN = 30
    OUTPUT_SEQ_LEN = 15

    FEATURE_DIM = 1
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    OUTPUT_DIM = 1
    NUM_LAYERS = 4

    TEACHER_FORCING_RATIO = 0.5
