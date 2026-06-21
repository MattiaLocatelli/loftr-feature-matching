import cv2
import kornia as K
import torch
import os
from kornia.feature import LoFTR
from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt

# Configurazione dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento LoFTR
matcher = LoFTR(pretrained="outdoor").to(device)

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return K.image_to_tensor(img, keepdim=False).float().to(device) / 255.0

# 1. Setup
online_img = load_img("./Online_Keyframe/third_billboard6.png")
offline_folder = "./Offline_Keyframes/"
offline_images = [f for f in os.listdir(offline_folder) if f.endswith('.png')]

results = []

# 2. Pipeline
valid_matches = []
MIN_INLIERS = 30

for img_name in offline_images:
    offline_img = load_img(os.path.join(offline_folder, img_name))
    
    # Resizing per LoFTR (spesso necessario per performance/memoria)
    img0 = K.geometry.resize(online_img, (250, 960), antialias=True)
    img1 = K.geometry.resize(offline_img, (250, 960), antialias=True)
    # img0 = online_img
    # img1 = offline_img
    
    # # Aggiungi questo test dopo il caricamento delle immagini
    # input_dict = {"image0": K.color.rgb_to_grayscale(img0), "image1": K.color.rgb_to_grayscale(img0)}
    # with torch.inference_mode():
    #     test_match = matcher(input_dict)
    # print(f"Test self-matching: {test_match['keypoints0'].shape[0]} punti trovati.")
    
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img0),
        "image1": K.color.rgb_to_grayscale(img1),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    
    # 3. Validazione geometrica (Filtra i match spuri)
    if len(mkpts0) > 8:
        _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 1000)
        inliers_mask = inliers.flatten() > 0
        num_inliers = sum(inliers_mask)
        
        print(f"Immagine: {img_name} | Inliers: {num_inliers}")
        
        if num_inliers >= MIN_INLIERS:
            # Salviamo tutto il necessario per la visualizzazione
            valid_matches.append({
                "name": img_name,
                "inliers_count": num_inliers,
                "img0": img0,
                "img1": img1,
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
                "mask": inliers_mask
            })

# 4. Selezione migliori match
for match in valid_matches:
    print(f"\nVisualizzazione match per: {match['name']} ({match['inliers_count']} inliers)")
    
    draw_LAF_matches(
        K.feature.laf_from_center_scale_ori(torch.from_numpy(match['mkpts0']).view(1, -1, 2)),
        K.feature.laf_from_center_scale_ori(torch.from_numpy(match['mkpts1']).view(1, -1, 2)),
        torch.arange(len(match['mkpts0'])).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(match['img0']),
        K.tensor_to_image(match['img1']),
        match['mask'],
        draw_dict={"inlier_color": (0.1, 1, 0.1, 0.5), "tentative_color": None}
    )
    plt.title(f"Match: {match['name']} - Inliers: {match['inliers_count']}")
    plt.show()

if not valid_matches:
    print("Nessun match trovato sopra la soglia richiesta.")