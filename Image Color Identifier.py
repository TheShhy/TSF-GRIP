#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

def get_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_colors(image, num_colors, show_chart=True):
    resized_img = cv2.resize(image, (800, 600))
    reshaped_img = resized_img.reshape((resized_img.shape[0] * resized_img.shape[1], 3))

    clf = KMeans(n_clusters=num_colors)
    labels = clf.fit_predict(reshaped_img)

    counts = Counter(labels)
    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i] for i in range(num_colors)]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in range(num_colors)]
    rgb_colors = [ordered_colors[i] for i in range(num_colors)]

    if show_chart:
        plot_comparison(image, hex_colors, counts)

    return rgb_colors, hex_colors, counts

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def plot_comparison(image, colors, counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Original Image')

    wedges, texts, autotexts = ax2.pie(counts.values(), labels=colors, colors=colors, autopct='%1.1f%%')
    ax2.set_title('Color Distribution')

    for text, autotext in zip(texts, autotexts):
        text.set(size=8)
        autotext.set(size=8, color='white', fontweight='bold')

    plt.show()

# Example usage:
image_path = "Test 1.jpg"
image = get_image(image_path)
num_colors = 15
rgb_colors, hex_colors, color_distribution = get_colors(image, num_colors)


# In[ ]:




