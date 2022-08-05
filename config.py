import os
import torch

debug = 1

# gpu
ngpu = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate data
data_root = ['./datset/', 'C://Users/Uttam kumar/Downloads/CRVD_dataset']
output_root = 'C://Users/Uttam kumar/emvd_code/result/'

image_height                    = 128
image_width                     = 128
batch_size                      = 16
frame_num					    = 25
num_workers                     = 4


BLACK_LEVEL = 64
VALID_VALUE = 959


# log
model_name						          = 'C://Users/Uttam kumar/emvd_code/model/'
debug_dir						            = os.path.join('debug')
log_dir                         = os.path.join(debug_dir, 'log')
log_step                        = 10                                        # save the train log per log_step

# model store
model_save_root                       = os.path.join(model_name, 'model.pth')
best_model_save_root                  = os.path.join(model_name, 'model_best.pth')

# pretrained model path
checkpoint       = None if not os.path.exists(os.path.join(model_name, 'model.pth')) else os.path.join(model_name, 'model.pth')
start_epoch			 = 0
start_iter			 = 0

# parameter of train
learning_rate                   = 0.01
epoch                           = int(1e2)

# validation
valid_start_iter				= 10
valid_step              = 5
vis_data						    = 1 	# whether to visualize noisy and gt data
# clip threshold
image_min_value                 = 0
image_max_value                 = 1
image_norm_value                = 1

label_min_value                 = 0
label_max_value                 = 255
label_norm_value                = 1


