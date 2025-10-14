def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class RehabPile():
    def __init__(self):
        super(RehabPile, self)
        self.scenarios = [("test", "target")]
        self.class_names = ['walk', 'upstairs',]
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        # self.input_channels = 66
        # self.num_classes = 2

        # CNN and RESNET features
        self.final_out_channels = 32
        self.features_len = 1

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128
        
        
        
        
class UIPRMD_clf_bn_DS():
    def __init__(self):
        super(UIPRMD_clf_bn_DS, self)
        self.scenarios = [("test", "target")]
        self.class_names = ['walk', 'upstairs',]
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 66
        self.num_classes = 2

        # CNN and RESNET features
        self.final_out_channels = 32
        self.features_len = 1

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128




class IRDS_clf_bn_EFL():
    def __init__(self):
        super(IRDS_clf_bn_EFL, self)
        self.scenarios = [("test", "target")]
        self.class_names = ['walk', 'upstairs',]
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 66
        self.num_classes = 2

        # CNN and RESNET features
        self.final_out_channels = 32
        self.features_len = 1

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128




class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"), ("18", "27"), ("20", "5"), ("24", "8"), ("28", "27"), ("30", "20")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 32
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128
        