
# binary change detection
python main.py --model_name TFIGR  --data_name LEVIR+ --dataloader_name bs_8

python main.py --model_name A2NetBCD  --data_name LEVIR+ --dataloader_name bs_8

# semantic change detection
python main.py --model_name A2Net  --data_name SECOND --dataloader_name bs_8

python main.py --model_name A2Net34  --data_name SECOND --dataloader_name bs_8

python main.py --model_name SSCDL  --data_name SECOND --dataloader_name bs_8

python main.py --model_name TED  --data_name SECOND --dataloader_name bs_8

python main.py --model_name BiSRNet  --data_name SECOND --dataloader_name bs_8

python main.py --model_name SCanNet  --data_name SECOND --dataloader_name bs_8

# building damage assessment
python main.py --model_name ChangeOS  --data_name xview2 --dataloader_name bs_8

python main.py --model_name ChangeOS-GRM  --data_name xview2 --dataloader_name bs_8

# binary change detection
python main.py --model_name TFIGR  --data_name LEVIR --dataloader_name bs_32

python main.py --model_name TFIGR  --data_name SYSU --dataloader_name bs_32

python main.py --model_name A2NetBCD  --data_name LEVIR --dataloader_name bs_32

python main.py --model_name A2NetBCD  --data_name SYSU --dataloader_name bs_32
