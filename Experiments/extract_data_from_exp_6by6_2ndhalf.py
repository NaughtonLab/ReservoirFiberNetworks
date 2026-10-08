import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from sklearn.cluster import KMeans

def extract_frames_from_video(filepath, framepath, vid_fps, frames_extracted):
    vdo = cv.VideoCapture(filepath)
    if not vdo.isOpened():
        print("Error: Could not open video.")
    else:
        fps = vdo.get(cv.CAP_PROP_FPS)
        fps = np.rint(fps).astype(int)
        print(f"FPS of the video: {fps}")
        frame_rate_for_img = np.rint(fps/vid_fps).astype(int)
        print(fps, frame_rate_for_img)
        count = 0
        success = 1
        if not frames_extracted:
            while success:
                success, frame = vdo.read()
                if success:
                    count += 1
                    cv.imwrite(f"{framepath}/{count}.jpg", frame)
                else:
                    break

        return frame_rate_for_img

def get_crop_indices(frame, start, crop_indices=None):
    if crop_indices is None:
        crop_indices = [0, frame.shape[0], 0, frame.shape[1]]
        new_frame = frame[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]
    else:
        crop_indices = crop_indices
        new_frame = frame[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]   
    # plt.imshow(new_frame)
    # plt.show()

    y1, y2, x1, x2 = input("Enter the crop indices (y1, y2, x1, x2): ").split(",")
    crop_indices = [int(y1), int(y2), int(x1), int(x2)]
    new_frame = frame[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]
    # plt.imshow(new_frame)
    # plt.show()
    crop = input("Do you want to keep these crop indices? (y/n): ")
    if crop == "n":
        crop_indices = get_crop_indices(frame, start, crop_indices)
    else:
        crop_indices = crop_indices
    # plt.close()
    return crop_indices

def get_threshold_values(cropped_frame, threshold1, threshold2):

    thresh_indices = cropped_frame > threshold1
    threshold = thresh_indices*cropped_frame

    r_channel = threshold[:, :, 0]
    g_channel = threshold[:, :, 1]
    b_channel = threshold[:, :, 2]
    diff = r_channel-g_channel-b_channel ## pixels that have higher reds than greens and blues
    diff2 = diff-g_channel-b_channel
    diff3 = diff2-g_channel-b_channel
    if threshold2 == 0:
        diff4 = diff3
    else:
        diff4 = diff3*(diff3>threshold2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    noise = cv.morphologyEx(diff4, cv.MORPH_OPEN, kernel, iterations=2)

    inv = np.invert(noise)
    seg_img = np.array([noise.T, 0*inv.T, 0*inv.T]).T

    # plt.figure(figsize=(20, 10))
    # plt.subplot(241)
    # plt.imshow(threshold)
    # plt.title("Thresholded Image")
    # plt.subplot(242)
    # plt.imshow(diff)
    # plt.title("Difference Image")
    # plt.subplot(243)
    # plt.imshow(diff2)
    # plt.title("Difference Image 2")
    # plt.subplot(244)
    # plt.imshow(diff3)
    # plt.title("Difference Image 3")
    # plt.subplot(245)
    # plt.imshow(diff4)
    # plt.title("Difference Image 4")
    # plt.subplot(246)
    # plt.imshow(noise)
    # plt.title("Noise Image")
    # plt.subplot(247)
    # plt.imshow(inv)
    # plt.title("Inverted Noise Image")
    # plt.subplot(248)
    # plt.imshow(seg_img)
    # plt.title("Segmented Image")
    # plt.tight_layout()
    # plt.show()

    keep = input(f"Do you want to keep these threshold values ({threshold1, threshold2})? (y/n): ")
    if keep == "n":
        threshold1 = int(input("Enter the new threshold1 value: "))
        threshold2 = int(input("Enter the new threshold2 value: "))
        threshold1, threshold2 = get_threshold_values(cropped_frame, threshold1, threshold2)
    else:
        threshold1 = threshold1
        threshold2 = threshold2

    # plt.close()
    return threshold1, threshold2
    
def segment_frames(framepath, start, end, crop_indices, frame_rate_for_img, threshold1, threshold2):
    noise_list = []
    inv_list = []
    segmented_3ch = []
    for i in range(start, end+frame_rate_for_img, frame_rate_for_img):
        frame = cv.imread(f"{framepath}/{i}.jpg")
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = frame[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]

        thresh_indices = frame > threshold1
        threshold = thresh_indices*frame

        r_channel = threshold[:, :, 0]
        g_channel = threshold[:, :, 1]
        b_channel = threshold[:, :, 2]
        diff = r_channel-g_channel-b_channel ## pixels that have higher reds than greens and blues
        diff2 = diff-g_channel-b_channel
        diff3 = diff2-g_channel-b_channel
        diff4 = diff3*(diff3>threshold2)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        noise = cv.morphologyEx(diff4, cv.MORPH_OPEN, kernel, iterations=2)
        noise_list.append(noise)

        # inv = np.invert(noise)
        # # inv_list.append(inv)
        # seg_img = np.array([noise.T, 0*inv.T, 0*inv.T]).T

        # segmented_3ch.append(seg_img)

    return noise_list#, segmented_3ch

def get_point_positions(noise_list, threshold1, threshold2, area1, area2, find_area_thresholds):
    point_pos = np.zeros((200, 2, len(noise_list)))
    for j in range(len(noise_list)):
        img = noise_list[j]
        new_img = img*0
        if threshold2 == 0:
            new_img[img > threshold1] = 255
        else:
            new_img[img > threshold2] = 255

        analysis = cv.connectedComponentsWithStats(new_img, cv.CV_32S, connectivity=8)
        (totalLabels, label_ids, values, centroid) = analysis

        m = 0
        for i in range(totalLabels):
            area = values[i, cv.CC_STAT_AREA]
            if area1 < area < area2:
                (x, y) = centroid[i]
                point_pos[m, 0, j] = x 
                point_pos[m, 1, j] = y
                m = m+1

    if find_area_thresholds:
        plt.figure(figsize=(5, 5))
        for i in range(totalLabels):
            plt.scatter(point_pos[i, 0, ...].T, point_pos[i, 1, ...].T, s=5)
        plt.gca().invert_yaxis()
        plt.ylabel("Y position (px)")
        plt.xlabel("X position (px)")
        plt.title("Trajectory of points")

        plt.show()
        plt.close()

        keep = input("Do you want to keep these area thresholds? (y/n): ")
        if keep == "n":
            area1 = int(input("Enter the new area1 value: "))
            area2 = int(input("Enter the new area2 value: "))
            point_pos = get_point_positions(noise_list, threshold1, threshold2, area1, area2, find_area_thresholds)
        else:
            point_pos = point_pos

    return point_pos

def get_pixel_to_mm_conversion_factor(data, points):
    conversion_list = []
    for i in range(len(points)):
        data11 = data[data['Label'] == points[i][0]]
        data12 = data[data['Label'] == points[i][1]]
        print(points[i][0])
        print(data11)
        print(points[i][1])
        print(data12)

        if abs(data11['Y'].iloc[0]-data12['Y'].iloc[0]) <= 5:
            data11.iloc[0, 2] = data12.iloc[0, 2]
            distance1_px = abs(data11['X'].iloc[0]-data12['X'].iloc[0])
        elif abs(data11['X'].iloc[0]-data12['X'].iloc[0]) <= 5:
            data11.iloc[0, 1] = data12.iloc[0, 1]
            distance1_px = abs(data11['Y'].iloc[0]-data12['Y'].iloc[0])
        else:
            distance1_px = np.sqrt((data11['X'].iloc[0]-data12['X'].iloc[0])**2 + (data11['Y'].iloc[0]-data12['Y'].iloc[0])**2)            
        conversion_list.append(points[i][2]/distance1_px)
    conversion_factor = np.mean(conversion_list)
    return conversion_factor

def clean_and_convert_data_to_mm(data, conversion_factor, start, end, num_labelled_frames):
    min_label = data['Label'].min()
    max_label = data['Label'].max()
    # count = 0
    # store_time = 555555
    # store_label = 555555
    # for i in range(num_labelled_frames): # for each frame i.e for each instance in time
    #     for j in range(min_label, max_label+1):
    #         row_idx_j = np.where((data['Time']==i) & (data['Label']==j))[0]
    #         if row_idx_j.size == 0:
    #             print(f"Label {j} not found in frame {i}")
    #             continue
    #         else:
    #             x_j = data.iloc[row_idx_j[0], 1]
    #             y_j = data.iloc[row_idx_j[0], 2]
    #             for k in range(min_label, max_label+1):
    #                 if k == j:
    #                     continue
    #                 else:
    #                     row_idx_k = np.where((data['Time']==i) & (data['Label']==k))[0]
    #                     if row_idx_k.size==0:
    #                         if store_time == 555555 and store_label == 555555: # create a new data frame for Nan values on the first occurrence
    #                             print(f"Creating new Nan df for label {k} at time {i}")
    #                             Nan_df = pd.DataFrame({'Label':[k], 'X':[np.nan], 'Y':[np.nan], 'Time':[i]})
    #                             count += 1
    #                             store_time = i
    #                             store_label = k
    #                         elif store_time == i and store_label == k:
    #                             print(f"Not updating Nan df for label {k} at time {i}")
    #                             continue
    #                         else:
    #                             print("Updating Nan df", i, j, k, row_idx_k)
    #                             new_row = pd.DataFrame({'Label':[k], 'X':[np.nan], 'Y':[np.nan], 'Time':[i]})
    #                             Nan_df = pd.concat([Nan_df, new_row], ignore_index=True)
    #                             count += 1
    #                             store_time = i
    #                             store_label = k
    #                     else:
    #                         x_k = data.iloc[row_idx_k[0], 1]
    #                         y_k = data.iloc[row_idx_k[0], 2]

    #                         if abs(x_j-x_k) <= 5:
    #                             data.iloc[row_idx_k[0], 1] = x_j
    #                         elif abs(y_j-y_k) <= 5:
    #                             data.iloc[row_idx_k[0], 2] = y_j
    #                         else:
    #                             continue

    # data = pd.concat([data, Nan_df], ignore_index=True)
    data = data.sort_values(by=['Label', 'Time'])
    data['Xmm'] = data['X']*conversion_factor
    data['Ymm'] = data['Y']*conversion_factor
    data['Time_s'] = data['Time']*(end-start)/(data['Time'].max()-data['Time'].min()) + start

    return data

if __name__ == "__main__":
    net_size = 6

    VIDEO_FOLDER = f"/projects/naughton/Apoorva/fiber_network_project/fiber_network/Experiments/SAGE/{net_size}by{net_size}/Videos"
    CSV_FILE = os.path.join(os.path.dirname(__file__), f"SAGE/logbook_to_extract_frames.csv")
    COLUMNS_FOR_NAME = ['sample_freq', 'eval_freq', 'ang']

    vid_df = pd.read_csv(CSV_FILE)
    vid_df = vid_df[vid_df['net_size'] == net_size]

    for i in range(len(vid_df)):
        vid_params = vid_df.iloc[i]
        name = f"{vid_df.iloc[i]['sample_freq']}_{vid_df.iloc[i]['eval_freq']}_{vid_df.iloc[i]['ang']}.MP4"
        filepath = os.path.join(VIDEO_FOLDER, name)
        n_req = net_size*net_size + 2 * (net_size+1) * net_size

        print(filepath, n_req)

        framepath = f"{VIDEO_FOLDER}/{name}_frames"
        if not os.path.exists(framepath):
            os.makedirs(framepath)
        
        vid_fps = int(119.88)

        frames = [int(f.split('.')[0]) for f in os.listdir(framepath) if f.endswith(".jpg")]
        start = min(frames)
        end = max(frames)
        start_sec = start / vid_fps
        end_sec = end / vid_fps
        crop_indices = [vid_params['y1'], vid_params['y2'], vid_params['x1'], vid_params['x2']]
        
        threshold1 = vid_params['threshold1'] #180 #200
        threshold2 = vid_params['threshold2'] #200 #230
        
        data = np.load(f"{framepath}/segmented_images.npz", allow_pickle=True)
        noise_list = data['noise_list']
        noise_list = noise_list[::2]
        print("Loaded binary segmented images from npz file")

        area1 = 10 #int(input("Enter the area1 value: ")) #
        area2 = 5000 #int(input("Enter the area2 value: ")) #
        find_area_thresholds = False

        point_pos = get_point_positions(noise_list, threshold1, threshold2, area1, area2, find_area_thresholds)

        plt.figure(figsize=(5, 5))
        for j in range(point_pos.shape[0]):
            plt.scatter(point_pos[j, 0, ...].T, point_pos[j, 1, ...].T, s=5)
        plt.gca().invert_yaxis()
        plt.ylabel("Y position (px)")
        plt.xlabel("X position (px)")
        plt.title("Trajectory of points")

        plt.show()

        # KMeans clustering
        n = n_req + 1 # Number of clusters

        inertias = []
        x = []
        y = []
        for j in range(len(noise_list)):
            for k in range(n):
                x.append(point_pos[k, 0, j])
                y.append(point_pos[k, 1, j])

        data = list(zip(x, y))
        kmeans = KMeans(n_clusters=n, random_state=2)
        kmeans.fit(data)
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, s=5, c=kmeans.labels_)
        plt.gca().invert_yaxis()
        plt.ylabel("Y position (px)")
        plt.xlabel("X position (px)")
        plt.title("Trajectory of points")
        plt.grid()
        plt.show()
        print(np.unique(kmeans.labels_))

        num_labels = len(np.unique(kmeans.labels_))
        num_labelled_frames = len(noise_list)

        data = pd.DataFrame(columns=['Label','X','Y','Time'])

        for k in range(num_labelled_frames):
            for j in range(num_labels-1):
                new_df = pd.DataFrame({'Label':[kmeans.labels_[j+k*(num_labels-1)]], 'X':[x[j+k*(num_labels-1)]], 'Y':[y[j+k*(num_labels-1)]], 'Time':[k]})
                data = pd.concat([data, new_df], ignore_index=True)

        print(data['Label'].unique())
        idx_origin = np.where((data['X'] == 0) & (data['Y'] == 0))[0]
        if len(idx_origin) > 0:
            remove_origin_label = np.unique(data.iloc[idx_origin]['Label'])[0]
            data = data[data['Label'] != remove_origin_label]
            n = n - 1

        data['Y'] = noise_list[0].shape[0] - data['Y'] # Invert Y axis
        data.to_csv(f"{framepath}/raw_data.csv", index=False) # Save the data to a CSV file

        # if n > n_req or n < n_req:
        #     print("Since the number of clusters is not equal to the number of required points, further processing should be done manually.")
            
        # else:

        #     plt.figure(figsize=(5, 5))
        #     plt.scatter(data['X'], data['Y'], c=data['Label'], s=5)
        #     # plt.gca().invert_yaxis()
        #     plt.ylabel("Y position (px)")
        #     plt.xlabel("X position (px)")
        #     plt.title("Position of Points")
        #     plt.grid()
        #     plt.show()

        #     y_ranges = input("Enter the y ranges (comma separated): ").split(",")
        #     y_ranges = [int(i) for i in y_ranges]
        #     x_ranges = input("Enter the x ranges (comma separated): ").split(",")
        #     x_ranges = [int(i) for i in x_ranges]

        #     j = 0
        #     for m in range(len(y_ranges)-1):
        #         for k in range(len(x_ranges)-1):
        #             idx = (data['X'] >= x_ranges[k]) & (data['X'] <= x_ranges[k+1]) & (data['Y'] >= y_ranges[m]) & (data['Y'] <= y_ranges[m+1])
        #             if len(data[idx]) > 0:
        #                 data.loc[idx, 'Label'] = j
        #                 j += 1

        #     time_0_data = data[data['Time'] == 0]
        #     print(time_0_data['Label'].unique())
        #     px_to_mm = float(input("Enter the pixel to mm conversion factor: "))
        #     if px_to_mm == 0:
        #         plt.figure(figsize=(5, 5))
        #         plt.scatter(time_0_data['X'], time_0_data['Y'], c=time_0_data['Label'], s=5)
        #         plt.ylabel("Y position (px)")
        #         plt.xlabel("X position (px)")
        #         plt.title("Position of Points at t=0s")
        #         plt.show()

        #         points = []
        #         for i in range(3):
        #             label1 = int(input(f"Enter the label of point 1 for set {i}: "))
        #             label2 = int(input(f"Enter the label of point 2 for set {i}: "))
        #             distance = int(input(f"Enter the distance in mm between points {label1} and {label2}: "))
        #             points.append((label1, label2, distance))

        #         px_to_mm = get_pixel_to_mm_conversion_factor(time_0_data, points)
        #         if px_to_mm is not None:
        #             print(f"Pixel to mm conversion factor: {px_to_mm}")
        #             px_to_mm = float(px_to_mm)
        #             clean_data = clean_and_convert_data_to_mm(data, px_to_mm, start_sec, end_sec, num_labelled_frames)
        #             clean_data.to_csv(f"{framepath}/clean_data.csv", index=False)
        #             print("Data cleaned and converted to mm. Saved as ", f"{framepath}/clean_data_more.csv")
        #     else:
        #         print(f"Pixel to mm conversion factor: {px_to_mm}")
        #         px_to_mm = float(px_to_mm)
        #         clean_data = clean_and_convert_data_to_mm(data, px_to_mm, start_sec, end_sec, num_labelled_frames)
        #         clean_data.to_csv(f"{framepath}/clean_data.csv", index=False)
        #         print("Data cleaned and converted to mm. Saved as ", f"{framepath}/clean_data_more.csv")

