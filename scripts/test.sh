
# binary change detection
python main.py --model_name TFIGR  --data_name LEVIR+ --dataloader_name bs_8 --is_train 0
python main.py --model_name A2NetBCD  --data_name LEVIR+ --dataloader_name bs_8 --is_train 0
python main.py --model_name ARCDNetBCD  --data_name LEVIR+ --dataloader_name bs_8 --is_train 0
python main.py --model_name ChangeStar  --data_name LEVIR+ --dataloader_name bs_8 --is_train 0

# semantic change detection
python main.py --model_name A2Net  --data_name SECOND --dataloader_name bs_8 --is_train 0
python main.py --model_name A2NetMvit  --data_name SECOND --dataloader_name bs_8 --is_train 0
python main.py --model_name A2Net34  --data_name SECOND --dataloader_name bs_8 --is_train 0
python main.py --model_name SSCDL  --data_name SECOND --dataloader_name bs_8 --is_train 0
python main.py --model_name TED  --data_name SECOND --dataloader_name bs_8 --is_train 0
python main.py --model_name BiSRNet  --data_name SECOND --dataloader_name bs_8 --is_train 0
python main.py --model_name SCanNet  --data_name SECOND --dataloader_name bs_8 --is_train 0

python main.py --model_name A2Net  --data_name LandsatSCD --dataloader_name bs_16 --is_train 0
python main.py --model_name A2NetMvit  --data_name LandsatSCD --dataloader_name bs_8 --is_train 0
python main.py --model_name A2Net34  --data_name LandsatSCD --dataloader_name bs_16 --is_train 0
python main.py --model_name SSCDL  --data_name LandsatSCD --dataloader_name bs_16 --is_train 0
python main.py --model_name TED  --data_name LandsatSCD --dataloader_name bs_16 --is_train 0
python main.py --model_name BiSRNet  --data_name LandsatSCD --dataloader_name bs_16 --is_train 0
python main.py --model_name SCanNet  --data_name LandsatSCD --dataloader_name bs_16 --is_train 0

# building damage assessment
python main.py --model_name ChangeOS  --data_name xview2 --dataloader_name bs_8 --is_train 0
python main.py --model_name ChangeOS-GRM  --data_name xview2 --dataloader_name bs_8 --is_train 0
python main.py --model_name ARCDNet  --data_name xview2 --dataloader_name bs_8 --is_train 0

# binary change detection
python main.py --model_name TFIGR  --data_name LEVIR --dataloader_name bs_32 --is_train 0
python main.py --model_name A2NetBCD  --data_name LEVIR --dataloader_name bs_32 --is_train 0
python main.py --model_name ARCDNetBCD  --data_name LEVIR --dataloader_name bs_32 --is_train 0
python main.py --model_name ChangeStar  --data_name LEVIR --dataloader_name bs_32 --is_train 0

python main.py --model_name TFIGR  --data_name SYSU --dataloader_name bs_32 --is_train 0
python main.py --model_name A2NetBCD  --data_name SYSU --dataloader_name bs_32 --is_train 0
python main.py --model_name ARCDNetBCD  --data_name SYSU --dataloader_name bs_32 --is_train 0
python main.py --model_name ChangeStar  --data_name SYSU --dataloader_name bs_32 --is_train 0

python main.py --model_name TFIGR  --data_name S2Looking --dataloader_name bs_8 --is_train 0
python main.py --model_name A2NetBCD  --data_name S2Looking --dataloader_name bs_8 --is_train 0
python main.py --model_name ARCDNetBCD  --data_name S2Looking --dataloader_name bs_8 --is_train 0
python main.py --model_name ChangeStar  --data_name S2Looking --dataloader_name bs_8 --is_train 0