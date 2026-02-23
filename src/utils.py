import argsparse
from config import Config
import datetime

def parse_args():
  date_str = datetime.datetime.now().strftime('%Y-%m-%d')

  parser = argsparse.ArgumentParser(description="DCEVAE Training and Testing Pipeline")

  parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
  parser.add_argument('--n_epochs', type=int, default=Config.N_EPOCHS)
  parser.add_argument('--lr', type=int, default=Config.LEARNING_RATE)
  parser.add_argument('--seed', type=int, default=Config.SEED)
  parser.add_argument('--uc_dim', type=int, default=Config.UC_DIM)
  parser.add_argument('--ud_dim', type=int, default=Config.UD_DIM)
  parser.add_argument('--h_dim', type=int, default=Config.H_DIM)
  parser.add_argument('--act_fn', type=int, default=Config.ACT_FN)
  parser.add_argument('--corr_a', type=int, default=Config.CORR_RECON_ALPHA)
  parser.add_argument('--desc_a', type=int, default=Config.DESC_RECON_ALPHA)
  parser.add_argument('--pred_a', type=int, default=Config.PRED_ALPHA)
  parser.add_argument('--fair_b', type=int, default=Config.FAIR_BETA)
  parser.add_argument('--tc_b', type=int, default=Config.TC_BETA)

  parser.add_argument('--exp_name', type=str, default=date_str)

  return parser.parse_args()