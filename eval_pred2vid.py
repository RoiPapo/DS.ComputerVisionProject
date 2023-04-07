import cv2
import os
from eval import get_labels_knoladge, get_labels_start_end_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
colors = ['magenta', 'green', 'cyan', 'yellow', 'pink', 'red']
classes = {f"G{i}": i for i in range(6)}


def plot_actions(pred, gt, index):
    y_label, y_start, y_end = get_labels_knoladge(gt)
    df_1 = pd.DataFrame({'start_time': y_start, 'end_time': y_end, 'label': y_label})
    df_1.end_time = df_1.end_time - 1

    p_label, p_start, p_end = get_labels_start_end_time(pred)
    df_2 = pd.DataFrame({'start_time': p_start, 'end_time': p_end, 'label': p_label})
    df_2.end_time = df_2.end_time - 1
    fig = plt.figure()

    for i in range(len(y_label)):
        plt.plot([df_1['start_time'][i], df_1['end_time'][i]], [1, 1], color=colors[classes[df_1['label'][i]]],
                 linewidth=15)

    for i in range(len(p_label)):
        plt.plot([df_2['start_time'][i], df_2['end_time'][i]], [0, 0], color=colors[df_2['label'][i]], linewidth=15)
    if index <= 110:
        index = 110
    plt.vlines(x=index-110, ymin=1.2, ymax=1.5,
               colors='darkblue')
    plt.ylim(-2, 3)
    plt.text(0, 0.57, "GT", fontsize=12,verticalalignment='center', transform= fig.transFigure)
    plt.text(0, 0.4, "Prediction", fontsize=12, transform= fig.transFigure)
    plt.axis("off")
    fig.canvas.draw()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image[100:320,:,:]

def parse_gt(gt):
    gts = []
    with open(gt, 'r') as f:
        for line in f.readlines():
            start, end, label = line.split()
            gts.extend([label for _ in range(int(end) - int(start) + 1)])
    return gts

def add_segmentation(image, index, label_path, gt_path):
    gestures_dict = {f"G{i}": i for i in range(6)}
    with open(label_path, 'r') as pred_file:
        with open(gt_path,'r') as gt_file:
            pred_data = [gestures_dict[row] for row in pred_file.readlines()[1].split()]
            gt_data = [row.strip() for row in gt_file.readlines()]
    segmentation = plot_actions(pred=pred_data,gt=gt_data,index = index)
    segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)
    im_v = cv2.vconcat([image, segmentation])
    return im_v

def draw_text(img, text,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(0, 0),
              font_scale=1,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(255, 255, 255)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return

def main():
    vids= ['P026_tissue1','P032_tissue1','P033_tissue1']
    gestures = {"no gesture": "G0",
                "needle passing": "G1",
                "pull the suture": "G2",
                "Instrument tie": "G3",
                "Lay the knot": "G4",
                "Cut the suture": "G5"}
    gestures = {v: k for k, v in gestures.items()}
    os.makedirs('LabeledVideos', exist_ok=True)

    for vid in vids:
        image_folder = f'/datashare/APAS/frames/{vid}_side'
        label_path = f"/home/student/computer_vision/DS.ComputerVisionProject/results/try Fvecs: Efficient6/fold2/Sample size 5_EfficienetB6/{vid}"

        gt_path = f"/datashare/APAS/transcriptions_gestures/{vid}.txt"

        video_name = f'LabeledVideos/{vid}.mp4'
        with open(label_path, 'r') as f:
            predictions = f.readlines()[1].split()
        gts = parse_gt(gt_path)
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(video_name, fourcc, 30, (width, 700))

        for index, (image, pred, gt) in tqdm(enumerate(zip(images, predictions, gts))):
            image = cv2.imread(os.path.join(image_folder, image))
            image_with_segmentation = add_segmentation(image, index, label_path, gt_path)
            draw_text(image_with_segmentation, "Pred:" + gestures[pred], pos=(0, 30 + 480), text_color=(0, 0, 255))
            draw_text(image_with_segmentation, "Ground truth: " + gestures[gt], pos=(0, 480), text_color=(0, 255, 0))
            video.write(image_with_segmentation)

        cv2.destroyAllWindows()
        video.release()


if __name__ == '__main__':
    main()
