
from pprint import pprint
from warnings import warn


def base_opts(parser):
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--target_name", type=str, required=True)

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--imgdir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='saves')
    parser.add_argument("--source_lang", type=str, default='')
    parser.add_argument("--target_lang", type=str, default='')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--imgH", type=int, default=32)
    parser.add_argument("--nHidden", type=int, default=256)
    parser.add_argument("--nChannels", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--schedule", action='store_true')