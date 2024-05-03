import matplotlib.pyplot as plt


models = ['Nano (n)', 'Small (s)', 'Medium (m)', 'Large (l)', 'Ultra (x)']

box_p = [0.869, 0.872, 0.869, 0.858, 0.873]
recall = [0.892, 0.910, 0.909, 0.912, 0.885]
map50 = [0.922, 0.924, 0.933, 0.928, 0.926]
map50_95 = [0.699, 0.703, 0.726, 0.722, 0.715]


plt.figure(figsize=(10, 6))
plt.plot(models, box_p, marker='o', label='Box P')
plt.plot(models, recall, marker='o', label='Recall')
plt.plot(models, map50, marker='o', label='mAP50')
plt.plot(models, map50_95, marker='o', label='mAP50-95')

plt.xlabel('Models')
plt.ylabel('Performance Metrics')
plt.title('Performance Comparison of YOLO Models (All Classes)')
plt.legend()
plt.grid(True)
plt.show()

##########################################################3

box_p = [0.869, 0.872, 0.869, 0.858, 0.873]
recall = [0.892, 0.910, 0.909, 0.912, 0.885]
map50 = [0.922, 0.924, 0.933, 0.928, 0.926]
map50_95 = [0.699, 0.703, 0.726, 0.722, 0.715]


plt.figure(figsize=(10, 6))
plt.plot(models, box_p, marker='o', label='Box P')
plt.plot(models, recall, marker='o', label='Recall')
plt.plot(models, map50, marker='o', label='mAP50')
plt.plot(models, map50_95, marker='o', label='mAP50-95')

plt.xlabel('Models')
plt.ylabel('Performance Metrics')
plt.title('Performance Comparison of YOLO Models (Cricket Ball Class)')
plt.legend()
plt.grid(True)
plt.show()
