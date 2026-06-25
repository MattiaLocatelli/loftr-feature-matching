import cv2
import kornia as K
import torch
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import bisect
import numpy as np
from kornia.feature import LoFTR
from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoFTR load
matcher = LoFTR(pretrained="outdoor").to(device)

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return K.image.image_to_tensor(img, keepdim=False).float().to(device) / 255.0

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    
    fig, axes = plt.subplots(2, 1, figsize=(19, 9), dpi=dpi)
    
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

# 1. Setup
online_img = load_img("./Online_Keyframe/R1257.png")
offline_folder = "./Offline_Keyframes_Turn2-3/"
offline_images = [f for f in os.listdir(offline_folder) if f.endswith('.png')]

output_dir = "output_matches"
os.makedirs(output_dir, exist_ok=True)

results = []

# 2. Pipeline
valid_matches = []
inference_times = []
MIN_INLIERS = 0
CONFIDENCE_THRESHOLD = 0.7

for img_name in offline_images:
    offline_img = load_img(os.path.join(offline_folder, img_name))
    
    # Resizing due to GPU memory limits
    img0 = K.geometry.resize(online_img, (256, 960), antialias=True)
    img1 = K.geometry.resize(offline_img, (256, 960), antialias=True)
    # img0 = online_img
    # img1 = offline_img
    
    # # Test after images load
    # input_dict = {"image0": K.color.rgb_to_grayscale(img0), "image1": K.color.rgb_to_grayscale(img0)}
    # with torch.inference_mode():
    #     test_match = matcher(input_dict)
    # print(f"Test self-matching: {test_match['keypoints0'].shape[0]} points found.")
    
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img0),
        "image1": K.color.rgb_to_grayscale(img1),
    }

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    end_event.record()
    
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event)
    inference_times.append(inference_time)
    
    # Confidence filter
    conf = correspondences["confidence"].cpu().numpy()
    mask_conf = conf > CONFIDENCE_THRESHOLD
    
    mkpts0 = correspondences["keypoints0"].cpu().numpy()[mask_conf]
    mkpts1 = correspondences["keypoints1"].cpu().numpy()[mask_conf]
    
    # 3. Geometric validation
    if len(mkpts0) > 8:
        _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 1000)
        inliers_mask = inliers.flatten() > 0
        num_inliers = sum(inliers_mask)
        
        print(f"Frame: {img_name} | Inliers: {num_inliers} | Inf Time: {inference_time:.3f} ms")
        
        if num_inliers >= MIN_INLIERS:
            valid_matches.append({
                "name": img_name,
                "inliers_count": num_inliers,
                "img0": img0,
                "img1": img1,
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
                "mask": inliers_mask,
                "mask_conf": conf[mask_conf]
            })

# 4. Select best matches
for match in valid_matches:
    print(f"Visualization for match: {match['name']} ({match['inliers_count']} inliers)")
    
    # draw_LAF_matches(
    #     K.feature.laf_from_center_scale_ori(torch.from_numpy(match['mkpts0']).view(1, -1, 2)),
    #     K.feature.laf_from_center_scale_ori(torch.from_numpy(match['mkpts1']).view(1, -1, 2)),
    #     torch.arange(len(match['mkpts0'])).view(-1, 1).repeat(1, 2),
    #     K.tensor_to_image(match['img0']),
    #     K.tensor_to_image(match['img1']),
    #     match['mask'],
    #     draw_dict={"inlier_color": (0.1, 1, 0.1, 0.5), "tentative_color": None}
    # )
    # plt.title(f"Match: {match['name']} - Inliers: {match['inliers_count']}")
    
    img0_np = K.image.tensor_to_image(match['img0'].cpu())
    img1_np = K.image.tensor_to_image(match['img1'].cpu())
    
    # n_pts = len(match['mkpts0'])
    # color = np.array([[0.1, 1.0, 0.1, 0.5]] * n_pts)
    color = cm.jet(match['mask_conf'])
    
    text = ['LoFTR', 'Matches: {}'.format(len(mkpts0))]
    fig = make_matching_figure(img0_np, img1_np, match['mkpts0'], match['mkpts1'], color, text=text)

    save_path = os.path.join(output_dir, f"match_{match['name']}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close(fig)    

print(f"Mean Inference Time: {sum(inference_times)/len(inference_times):.3f}")

if not valid_matches:
    print("No match found above the threshold.")