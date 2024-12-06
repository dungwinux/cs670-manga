from pathlib import Path
from tqdm import tqdm
import glob
import torch
from fastai.vision import *
import fastai
from mts import MTS
import cv2
import numpy as np
import processing as proc
import algorithm as al
import argparse

manga109_dirs = [
    "AisazuNihaIrarenai",
    "AkkeraKanjinchou",
    "Akuhamu",
    "AosugiruHaru",
    "AppareKappore",
    "Arisa",
    "ARMS",
    "BakuretsuKungFuGirl",
    "Belmondo",
    "BEMADER_P",
    "BokuHaSitatakaKun",
    "BurariTessenTorimonocho",
    "ByebyeC-BOY",
    "Count3DeKimeteAgeru",
    "DollGun",
    "Donburakokko",
    "DualJustice",
    "EienNoWith",
    "EvaLady",
    "EverydayOsakanaChan",
    "GakuenNoise",
    "GarakutayaManta",
    "GinNoChimera",
    "GOOD_KISS_Ver2",
    "Hamlet",
    "HanzaiKousyouninMinegishiEitarou",
    "HaruichibanNoFukukoro",
    "HarukaRefrain",
    "HealingPlanet",
    "HeiseiJimen",
    "HighschoolKimengumi_vol01",
    "HighschoolKimengumi_vol20",
    "HinagikuKenzan",
    "HisokaReturns",
    "JangiriPonpon",
    "JijiBabaFight",
    "Joouari",
    "Jyovolley",
    "KarappoHighschool",
    "KimiHaBokuNoTaiyouDa",
    "KoukouNoHitotachi",
    "KuroidoGanka",
    "KyokugenCyclone",
    "LancelotFullThrottle",
    "LoveHina_vol01",
    "LoveHina_vol14",
    "MadouTaiga",
    "MAD_STONE",
    "MagicianLoad",
    "MagicStarGakuin",
    "MariaSamaNihaNaisyo",
    "MayaNoAkaiKutsu",
    "MemorySeijin",
    "MeteoSanStrikeDesu",
    "MiraiSan",
    "MisutenaideDaisy",
    "MoeruOnisan_vol01",
    "MoeruOnisan_vol19",
    "MomoyamaHaikagura",
    "MukoukizuNoChonbo",
    "MutekiBoukenSyakuma",
    "Nekodama",
    "NichijouSoup",
    "Ningyoushi",
    "OhWareraRettouSeitokai",
    "OL_Lunch",
    "ParaisoRoad",
    "PikaruGenkiDesu",
    "PLANET7",
    "PlatinumJungle",
    "PrayerHaNemurenai",
    "PrismHeart",
    "PsychoStaff",
    "Raphael",
    "ReveryEarth",
    "RinToSiteSippuNoNaka",
    "RisingGirl",
    "Saisoku",
    "SaladDays_vol01",
    "SaladDays_vol18",
    "SamayoeruSyonenNiJunaiWo",
    "SeisinkiVulnus",
    "ShimatteIkouze_vol01",
    "ShimatteIkouze_vol26",
    "SonokiDeABC",
    "SyabondamaKieta",
    "TaiyouNiSmash",
    "TapkunNoTanteisitsu",
    "TasogareTsushin",
    "TennenSenshiG",
    "TensiNoHaneToAkumaNoShippo",
    "TetsuSan",
    "That'sIzumiko",
    "TotteokiNoABC",
    "ToutaMairimasu",
    "TouyouKidan",
    "TsubasaNoKioku",
    "UchiNoNyan'sDiary",
    "UchuKigekiM774",
    "UltraEleven",
    "UnbalanceTokyo",
    "WarewareHaOniDearu",
    "YamatoNoHane",
    "YasasiiAkuma",
    "YouchienBoueigumi",
    "YoumaKourin",
    "YukiNoFuruMachi",
    "YumeiroCooking",
    "YumeNoKayoiji",
]

def is_problematic(test: str):
    if test.startswith('PrayerHaNemurenai'):
        return True
    return False

def is_cached(cwd: Path, test: str):
    if (cwd / 'mts_caches' / test).with_suffix('.png').exists():
        return True
    return False

def prepare(cwd, dir_list, *, overwrite=False, try_cpu=False):
    # Since we are reusing the mask, let's cache them before cv2 operation
    input_path = cwd / 'inputs'
    output_path = cwd / 'mts_caches'
    filt = '/00*.jpg' # Select the first 10
    root = pathlib.Path(__file__).parents[0].resolve()
    # torch.cuda.set_device(0)

    # torch.cuda.empty_cache()
    # fastai.torch_core.defaults.device = torch.device('cpu')
    # fastai.torch_core.defaults.device = torch.device('cuda')
    
    for (i, learn) in MTS.m_learner.items():
        # print(learn.summary())
        # Individually load manually to avoid overloadding the GPU
        # learn = load_learner(str(root / MTS.model_path), f'fold.{i}.-.final.refined.model.2.pkl')
        test_set = []
        for d in dir_list:
            test_set.extend(glob.glob(d + filt, root_dir=input_path, recursive=True))
        for test in tqdm(test_set):
            # Exclude problematic ones
            if is_problematic(test):
                continue
            if overwrite or not (output_path / str(i) / test).with_suffix('.png').exists():
                if try_cpu:
                    print(f"Trying CPU once on {test}")
                    fastai.torch_core.defaults.device = torch.device('cpu')
                    proc.mts_cache(i, test, input_path, output_path)
                    return
                try:
                    proc.mts_cache(i, test, input_path, output_path)
                except torch.OutOfMemoryError:
                    print(f"GPU OOM while processing {test}. Skipping")

def old_main():
    import port
    cwd = Path('/home/saratoga/cs670-manga/')
    td = port.TextDetection(cwd)
    path_all = '**/*.jpg'
    path_openmantra = 'open-mantra-dataset/images/**/*.jpg'
    path_tsubasa_no_kioku = 'TsubasaNoKioku/**/*.jpg'
    test_set = glob.glob(path_all, root_dir=td.input_base, recursive=True)
    for test in tqdm(test_set):
        td.execute(test)

def main():
    parser = argparse.ArgumentParser(description='Batch processing at port')
    parser.add_argument('--er', action='store_true')
    parser.add_argument('--swt', action='store_true')
    parser.add_argument('--db', action='store_true')
    parser.add_argument('--east', action='store_true')
    parser.add_argument('--textboxes', action='store_true')
    parser.add_argument('--stub', action='store_true')
    args = parser.parse_args()

    cwd = Path('/home/saratoga/cs670-manga/')
    filt = '/00*.jpg'

    prepare(cwd, manga109_dirs)

    import port
    td = port.TextDetection(cwd)

    if args.stub:
        print('Running Stub algorithm')
        td.cv2_model = al.Stub()
    elif args.textboxes:
        print('Running CNNTextBoxes algorithm')
        td.cv2_model = al.CNNTextBoxes()
    elif args.east:
        print('Running EAST algorithm')
        td.cv2_model = al.DNN_EAST()
        al.enable_CUDA_if_available(td.cv2_model)
    elif args.db:
        print('Running DB algorithm')
        td.cv2_model = al.DNN_DB()
        al.enable_CUDA_if_available(td.cv2_model)
    elif args.swt:
        print('Running SWT algorithm')
        td.cv2_model = al.Text_SWT()
    elif args.er:
        print('Running ER algorithm')
        td.cv2_model = al.Text_ER()

    
    test_set = []
    for d in manga109_dirs:
        test_set.extend(glob.glob(d + filt, root_dir=td.input_base, recursive=True))
    
    for test in tqdm(test_set):
        if is_problematic(test):
            continue
        if not (td.output_base2 / test).with_suffix('.png').exists():
            td.execute(test, use_mts_cache=True)

with torch.inference_mode():
    main()
