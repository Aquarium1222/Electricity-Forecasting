class Constant:
    RESERVE_MARGIN = './data/每日尖峰備轉容量率.csv'
    RESERVE_MARGIN_TEST = './data/本年度每日尖峰備轉容量率.csv'
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
    START_DATE = '2022/03/27'


class Hyperparameter:
    LEARNING_RATE = 0.0002
    BATCH_SIZE = 16
    EPOCH = 100
    PATIENCE = 10

    INPUT_SEQ_LEN = 30
    OUTPUT_SEQ_LEN = 7

    FEATURE_DIM = 1
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    OUTPUT_DIM = 1
    NUM_LAYERS = 4

    TEACHER_FORCING_RATIO = 0.5
