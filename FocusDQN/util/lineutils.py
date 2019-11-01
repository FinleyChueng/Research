# import math
#
# def distribution_2angle(center_coord, regions):
#
#
#
#
#     return
#
#
#
#
# # 取图片，然后求解外接圆，再按照region的数量来平分圆。
#
# adapter = BratsAdapter('Train')
# for _ in range(35):
#     adapter.next_image_pair()
# _, a = adapter.next_image_pair()
# a.dtype = np.uint8
# a = cv2.resize(a, (30, 30))
#
# xlist, ylist = np.where(a == 1)
# minx = min(xlist)
# maxx = max(xlist)
# miny = min(ylist)
# maxy = max(ylist)
# print(xlist)
# print(ylist)
# print(minx, miny, maxx, maxy)
#
# ox = (minx + maxx) / 2
# oy = (miny + maxy) / 2
# # radius = np.sqrt((ox-minx)**2 + (oy-miny)**2)
# radius = (ox-minx)**2 + (oy-miny)**2
# print(ox, oy, radius)
#
# b = np.zeros(a.shape, dtype=np.uint8)
# for x in range(a.shape[0]):
#     for y in range(a.shape[1]):
#         # if (x - ox) ** 2 + (y - oy) ** 2 <= radius ** 2:
#         if (x - ox) ** 2 + (y - oy) ** 2 <= radius:
#             b[x, y] = 1
#
# print(a)
#
# print(b)
# # upper
# print(a+b)
#
# c = b.copy()
# # 腐蚀
# while True:
#     diff = a - c
#     if (diff[np.where(c == 1)] == 0).all():
#         break
#     # operate.
#     c = morphology.erosion(c)
#
# print(c)
# # lower
# print(a+c)
#
#
# print('-----------------  From Here  ---------------------')
#
# # 计算直线
# import math
#
# # regions = 6
# regions = 1
# per_angle = 360 // regions
#
# result = np.zeros((a.shape[0], a.shape[1], regions))
# print(b)
# for idx in range(regions):
#     d = b.copy()
#     for x in range(a.shape[0]):
#         for y in range(a.shape[1]):
#             # The flag indicating whether to set current pixel to zero.
#             cur_is_bg = False
#             # Compute the angle and the gradient of two line.
#             a1 = idx * per_angle
#             k1 = round(math.tan(math.radians(a1)), 6)
#             a2 = (idx+1) * per_angle
#             k2 = round(math.tan(math.radians(a2)), 6)
#             # The gradient is response to the tan(), so 180 degree is a watershed.
#             if a2 <= 360 // 2:
#                 # Judge from different situations.
#                 if k1 < 0 and k2 < 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 < 0 and k2 == 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 < 0 and k2 > 0:
#                     raise Exception('(0 < a2 <= 180) --> Impossible situation [k1({}) < 0 and k2({}) > 0] !!! '.format(
#                         k1, k2
#                     ))
#                 elif k1 == 0 and k2 < 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 == 0 and k2 == 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 == 0 and k2 > 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 > 0 and k2 < 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 > 0 and k2 == 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 > 0 and k2 > 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 else:
#                     raise Exception('Invalid gradient situation !!! --> (0 < a2 <= 180) k1: {}, k2: {}'.format(k1, k2))
#             # The angle is no more than 360 degree. 360 = a circle.
#             elif a2 <= 360:
#                 # Judge from different situations.
#                 if k1 < 0 and k2 < 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 < 0 and k2 == 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy < k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 < 0 and k2 > 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 == 0 and k2 < 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 == 0 and k2 == 0:
#                     # We should judge a situation that only slice whole circle into
#                     #   one region.
#                     if a1 != 0:
#                         # Set the outside pixels to zeros.
#                         if y - oy > k1 * (x - ox) or y - oy > k2 * (x - ox):
#                             cur_is_bg = True
#                     else:
#                         # Which means there is totally one region, so no conditions and
#                         #   simply preserve all pixels. Pass current pixel.
#                         pass
#                 elif k1 == 0 and k2 > 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 > 0 and k2 < 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 > 0 and k2 == 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy > k2 * (x - ox):
#                         cur_is_bg = True
#                 elif k1 > 0 and k2 > 0:
#                     # Set the outside pixels to zeros.
#                     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#                         cur_is_bg = True
#                 else:
#                     raise Exception('Invalid gradient situation !!! --> (180 < a2 <= 360) k1: {}, k2: {}'.format(k1, k2))
#             # Valid angle.
#             else:
#                 raise Exception('Angle is beyond the 360 degree !!! --> a1: {}, a2: {}'.format(a1, a2))
#
#             # Set the background according to the flag.
#             if cur_is_bg:
#                 d[x, y] = 0
#
#     # Console output.
#     print(d)
#     print(b + d)
#     result[:, :, idx] = d
#
# fusion = np.zeros(b.shape)
# for i in range(regions):
#     # Visualization.
#     win_id = 330 + i + 1
#     plt.subplot(win_id)
#     plt.imshow(result[:, :, i], cmap='gray')
#     # Fusion.
#     fusion[np.where(result[:, :, i] == 1)] = 1
# plt.subplot(338)
# plt.imshow(b, cmap='gray')
# plt.subplot(339)
# plt.imshow(fusion, cmap='gray')
# plt.show()
# print(math.tan(math.radians(1*per_angle)))
# print(math.tan(math.radians(2*per_angle)))
#
#
#
#
#
#
#
#
#
# # 将整体功能，划分成 几个函数， 因此， 优化了性能。
# #	而且，可以整合成一个单独的类。
#
#
#
# def upper_lower_circle_boundary(src):
#     xlist, ylist = np.where(src == 1)
#     minx = min(xlist)
#     maxx = max(xlist)
#     miny = min(ylist)
#     maxy = max(ylist)
#     print(xlist)
#     print(ylist)
#     print(minx, miny, maxx, maxy)
#
#     ox = (minx + maxx) / 2
#     oy = (miny + maxy) / 2
#     # radius = np.sqrt((ox-minx)**2 + (oy-miny)**2)
#     radius = (ox - minx) ** 2 + (oy - miny) ** 2
#     print(ox, oy, radius)
#
#     upper = np.zeros(a.shape, dtype=np.uint8)
#     for x in range(a.shape[0]):
#         for y in range(a.shape[1]):
#             if (x - ox) ** 2 + (y - oy) ** 2 <= radius:
#                 upper[x, y] = 1
#
#     print(src)
#
#     print(upper)
#     # upper
#     print(src + upper)
#
#     lower = upper.copy()
#     # 腐蚀
#     while True:
#         diff = src - lower
#         if (diff[np.where(lower == 1)] == 0).all():
#             break
#         # operate.
#         lower = morphology.erosion(lower)
#
#     print(lower)
#     # lower
#     print(src + lower)
#
#     return upper, lower, ox, oy
#
# def mask_specific_region(src, reg_idx, regions, ox, oy):
#     # The mask.
#     mask = src.copy()
#     # Calculate the average angle.
#     per_angle = 360 // regions
#     # Compute the angle and the gradient of two line.
#     a1 = reg_idx * per_angle
#     k1 = round(math.tan(math.radians(a1)), 6)
#     a2 = (reg_idx + 1) * per_angle
#     k2 = round(math.tan(math.radians(a2)), 6)
#     # Situation.
#     situation = None
#     # The gradient is response to the tan(), so 180 degree is a watershed.
#     if a2 <= 360 // 2:
#         # Judge from different situations.
#         if k1 < 0 and k2 < 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 < 0 and k2 == 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 < 0 and k2 > 0:
#             raise Exception(
#                 '(0 < a2 <= 180) --> Impossible situation [k1({}) < 0 and k2({}) > 0] !!! '.format(
#                     k1, k2
#                 ))
#         elif k1 == 0 and k2 < 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 == 0 and k2 == 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 == 0 and k2 > 0:
#             # Situation 3.
#             situation = 's3'
#         elif k1 > 0 and k2 < 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 > 0 and k2 == 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 > 0 and k2 > 0:
#             # Situation 3.
#             situation = 's3'
#         else:
#             raise Exception(
#                 'Invalid gradient situation !!! --> (0 < a2 <= 180) k1: {}, k2: {}'.format(k1, k2))
#     # The angle is no more than 360 degree. 360 = a circle.
#     elif a2 <= 360:
#         # Judge from different situations.
#         if k1 < 0 and k2 < 0:
#             # Situation 3.
#             situation = 's3'
#         elif k1 < 0 and k2 == 0:
#             # Situation 3.
#             situation = 's3'
#         elif k1 < 0 and k2 > 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 == 0 and k2 < 0:
#             # Situation 4.
#             situation = 's4'
#         elif k1 == 0 and k2 == 0:
#             # We should judge a situation that only slice whole circle into
#             #   one region.
#             if a1 != 0:
#                 # Situation 4.
#                 situation = 's4'
#             else:
#                 # Which means there is totally one region, so no conditions and
#                 #   simply preserve all pixels. Pass current pixel.
#                 return mask
#         elif k1 == 0 and k2 > 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 > 0 and k2 < 0:
#             # Situation 4.
#             situation = 's4'
#         elif k1 > 0 and k2 == 0:
#             # Situation 4.
#             situation = 's4'
#         elif k1 > 0 and k2 > 0:
#             # Situation 1.
#             situation = 's1'
#         else:
#             raise Exception(
#                 'Invalid gradient situation !!! --> (180 < a2 <= 360) k1: {}, k2: {}'.format(k1, k2))
#     # Valid angle.
#     else:
#         raise Exception('Angle is beyond the 360 degree !!! --> a1: {}, a2: {}'.format(a1, a2))
#
#     # Different range operation according to the different situation.
#     if situation == 's1':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_mk1_lk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     elif situation == 's2':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_lk1_lk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     elif situation == 's3':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_lk1_mk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     elif situation == 's4':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_mk1_mk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     else:
#         raise Exception('Invalid situation !!!')
#
#     # Finally get the mask.
#     return mask
#
# # Situation 1.
# def _line_mk1_lk2(x, y, ox, oy, k1, k2):
#     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#         return True
#     else:
#         return False
#
# # Situation 2.
# def _line_lk1_lk2(x, y, ox, oy, k1, k2):
#     if y - oy < k1 * (x - ox) or y - oy < k2 * (x - ox):
#         return True
#     else:
#         return False
#
# # Situation 3.
# def _line_lk1_mk2(x, y, ox, oy, k1, k2):
#     if y - oy < k1 * (x - ox) or y - oy > k2 * (x - ox):
#         return True
#     else:
#         return False
#
# # Situation 4.
# def _line_mk1_mk2(x, y, ox, oy, k1, k2):
#     if y - oy > k1 * (x - ox) or y - oy > k2 * (x - ox):
#         return True
#     else:
#         return False
#
#
# # 从这里开始，是调用，以及如何膨胀，收缩
#
# regions = 8
# result = np.zeros((a.shape[0], a.shape[1], regions))
#
# start = time.time()
# # first get the upper and lower bound.
# b, c, ox, oy = upper_lower_circle_boundary(a)
# print(time.time() - start)
#
# # start = time.time()
# # d = mask_specific_region(b, 0, regions, ox, oy)
# # print(time.time() - start)
#
# start = time.time()
# # get each region.
# for idx in range(regions):
#     d = mask_specific_region(b, idx, regions, ox, oy)
#     # Dilation the specific region.
#     if idx == 2:
#         for _ in range(7):
#             d = morphology.dilation(d)
#         d = mask_specific_region(d, idx, regions, ox, oy)
#     # Erosion the specific region.
#     if idx == 5:
#         # Dilation the opposition is the same function.
#         template = np.ones(d.shape, dtype=np.uint8)
#         cur_reg = mask_specific_region(template, idx, regions, ox, oy)
#         opposition = cur_reg - d
#         for _ in range(7):
#             opposition = morphology.dilation(opposition)
#         opposition = mask_specific_region(opposition, idx, regions, ox, oy)
#         d = cur_reg - opposition
#     # Console output.
#     print(d)
#     print(b + d)
#     result[:, :, idx] = d
# print(time.time() - start)
#
#
# fusion = np.zeros(b.shape)
# for i in range(regions):
#     # Visualization.
#     win_id = 330 + i + 1
#     plt.subplot(win_id)
#     plt.imshow(result[:, :, i], cmap='gray')
#     # Fusion.
#     fusion[np.where(result[:, :, i] == 1)] = 1
# plt.subplot(338)
# plt.imshow(b, cmap='gray')
# plt.subplot(339)
# plt.imshow(fusion, cmap='gray')
# plt.show()
#
#
#
#
#
#
# # 18/10/19 22:30   优化了函数，而且能够成功合理地腐蚀和膨胀。 还完善了 腐蚀 和 膨胀 的 终止逻辑。
#
#
#
# # Matrix Mat. ------------------------------------------------
#
# import math
#
#
# def upper_lower_circle_boundary(src):
#     xlist, ylist = np.where(src == 1)
#     minx = min(xlist)
#     maxx = max(xlist)
#     miny = min(ylist)
#     maxy = max(ylist)
#     print(xlist)
#     print(ylist)
#     print(minx, miny, maxx, maxy)
#
#     ox = (minx + maxx) / 2
#     oy = (miny + maxy) / 2
#     # radius = np.sqrt((ox-minx)**2 + (oy-miny)**2)
#     radius = (ox - minx) ** 2 + (oy - miny) ** 2
#     print(ox, oy, radius)
#
#     upper = np.zeros(a.shape, dtype=np.uint8)
#     for x in range(a.shape[0]):
#         for y in range(a.shape[1]):
#             if (x - ox) ** 2 + (y - oy) ** 2 <= radius:
#                 upper[x, y] = 1
#
#     # print(src)
#     #
#     # print(upper)
#     # # upper
#     # print(src + upper)
#
#     lower = upper.copy()
#     # 腐蚀
#     while True:
#         diff = src - lower
#         if (diff[np.where(lower == 1)] == 0).all():
#             break
#         # operate.
#         lower = morphology.erosion(lower, selem=morphology.disk(5))
#         # lower = morphology.erosion(lower)
#
#     # print(lower)
#     # # lower
#     # print(src + lower)
#
#     return upper, lower, ox, oy
#
# def mask_specific_region(src, reg_idx, regions, ox, oy):
#     # The mask.
#     mask = src.copy()
#     # Calculate the average angle.
#     per_angle = 360 // regions
#     # Compute the angle and the gradient of two line.
#     a1 = reg_idx * per_angle
#     # k1 = round(math.tan(math.radians(a1)), 6)
#     k1 = round(math.tan(math.radians(a1)), 10)
#     a2 = (reg_idx + 1) * per_angle
#     # k2 = round(math.tan(math.radians(a2)), 6)
#     k2 = round(math.tan(math.radians(a2)), 10)
#     # Situation.
#     situation = None
#     # The gradient is response to the tan(), so 180 degree is a watershed.
#     if a2 <= 360 // 2:
#         # Judge from different situations.
#         if k1 < 0 and k2 < 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 < 0 and k2 == 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 < 0 and k2 > 0:
#             raise Exception(
#                 '(0 < a2 <= 180) --> Impossible situation [k1({}) < 0 and k2({}) > 0] !!! '.format(
#                     k1, k2
#                 ))
#         elif k1 == 0 and k2 < 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 == 0 and k2 == 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 == 0 and k2 > 0:
#             # Situation 3.
#             situation = 's3'
#         elif k1 > 0 and k2 < 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 > 0 and k2 == 0:
#             # Situation 2.
#             situation = 's2'
#         elif k1 > 0 and k2 > 0:
#             # Situation 3.
#             situation = 's3'
#         else:
#             raise Exception(
#                 'Invalid gradient situation !!! --> (0 < a2 <= 180) k1: {}, k2: {}'.format(k1, k2))
#     # The angle is no more than 360 degree. 360 = a circle.
#     elif a2 <= 360:
#         # Judge from different situations.
#         if k1 < 0 and k2 < 0:
#             # Situation 3.
#             situation = 's3'
#         elif k1 < 0 and k2 == 0:
#             # Situation 3.
#             situation = 's3'
#         elif k1 < 0 and k2 > 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 == 0 and k2 < 0:
#             # Situation 4.
#             situation = 's4'
#         elif k1 == 0 and k2 == 0:
#             # We should judge a situation that only slice whole circle into
#             #   one region.
#             if a1 != 0:
#                 # Situation 4.
#                 situation = 's4'
#             else:
#                 # Which means there is totally one region, so no conditions and
#                 #   simply preserve all pixels. Pass current pixel.
#                 return mask
#         elif k1 == 0 and k2 > 0:
#             # Situation 1.
#             situation = 's1'
#         elif k1 > 0 and k2 < 0:
#             # Situation 4.
#             situation = 's4'
#         elif k1 > 0 and k2 == 0:
#             # Situation 4.
#             situation = 's4'
#         elif k1 > 0 and k2 > 0:
#             # Situation 1.
#             situation = 's1'
#         else:
#             raise Exception(
#                 'Invalid gradient situation !!! --> (180 < a2 <= 360) k1: {}, k2: {}'.format(k1, k2))
#     # Valid angle.
#     else:
#         raise Exception('Angle is beyond the 360 degree !!! --> a1: {}, a2: {}'.format(a1, a2))
#
#     # Different range operation according to the different situation.
#     if situation == 's1':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_mk1_lk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     elif situation == 's2':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_lk1_lk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     elif situation == 's3':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_lk1_mk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     elif situation == 's4':
#         # Recursively judge the affiliation of each
#         for x in range(src.shape[0]):
#             for y in range(src.shape[1]):
#                 # Judge whether current pixel is background.
#                 if _line_mk1_mk2(x, y, ox, oy, k1, k2):
#                     # Set the background according to the flag.
#                     mask[x, y] = 0
#     else:
#         raise Exception('Invalid situation !!!')
#
#     # Finally get the mask.
#     return mask
#
# # Situation 1.
# def _line_mk1_lk2(x, y, ox, oy, k1, k2):
#     if y - oy > k1 * (x - ox) or y - oy < k2 * (x - ox):
#         return True
#     else:
#         return False
#
# # Situation 2.
# def _line_lk1_lk2(x, y, ox, oy, k1, k2):
#     if y - oy < k1 * (x - ox) or y - oy < k2 * (x - ox):
#         return True
#     else:
#         return False
#
# # Situation 3.
# def _line_lk1_mk2(x, y, ox, oy, k1, k2):
#     if y - oy < k1 * (x - ox) or y - oy > k2 * (x - ox):
#         return True
#     else:
#         return False
#
# # Situation 4.
# def _line_mk1_mk2(x, y, ox, oy, k1, k2):
#     if y - oy > k1 * (x - ox) or y - oy > k2 * (x - ox):
#         return True
#     else:
#         return False
#
#
#
# adapter = BratsAdapter('Train')
# for _ in range(45):
# # for _ in range(35):
#     adapter.next_image_pair()
# _, a = adapter.next_image_pair()
# a.dtype = np.uint8
# # a = cv2.resize(a, (30, 30))
# print(a)
#
# # regions = 12
# regions = 20
# # regions = 72
# result = np.zeros((a.shape[0], a.shape[1], regions))
#
# start = time.time()
# # first get the upper and lower bound.
# b, c, ox, oy = upper_lower_circle_boundary(a)
# # c, b, ox, oy = upper_lower_circle_boundary(a)
# print(time.time() - start)
#
# # start = time.time()
# # d = mask_specific_region(b, 0, regions, ox, oy)
# # print(time.time() - start)
#
# # Erosen to ground truth.
# start = time.time()
# res = b.copy()
# for reg_id in range(regions):
#     # Get original region.
#     org = mask_specific_region(a, reg_id, regions, ox, oy)
#     # Operation.
#     opt = 1
#     while True:
#         # get specific region.
#         d = mask_specific_region(res, reg_id, regions, ox, oy)
#         # Duplication.
#         be_d = d.copy()
#         # # Dilation the specific region.
#         # if opt == 0:
#         #     for _ in range(2):
#         #         # for _ in range(7):
#         #         d = morphology.dilation(d, selem=morphology.disk(5))
#         #     d = mask_specific_region(d, reg_id, regions, ox, oy)
#         # Erosion the specific region.
#         if opt == 1:
#             # Dilation the opposition is the same function.
#             template = np.ones(d.shape, dtype=np.uint8)
#             cur_reg = mask_specific_region(template, reg_id, regions, ox, oy)
#             opposition = cur_reg - d
#             for _ in range(1):
#             # for _ in range(7):
#                 opposition = morphology.dilation(opposition, selem=morphology.disk(5))
#             opposition = mask_specific_region(opposition, reg_id, regions, ox, oy)
#             d = cur_reg - opposition
#
#         # print('--- id = {}  ---'.format(reg_id))
#         # print(d)
#         # print(org)
#         # print(d-org)
#         # print('---------')
#
#         # Judge whether terminated or not.
#         diff = d - org
#         if (diff[np.where(org == 1)] != 0).any():
#         # if not (diff[np.where(org == 1)] == 0).all():
#             break
#         # Erase the selected region.
#         res -= be_d
#         # Record the operation. (Add the region that previously erased)
#         res += d
# print(time.time() - start)
#
# # Dilation to the ground truth.
# start = time.time()
# res_dil = c.copy()
# for reg_id in range(regions):
#     # Get original region.
#     org = mask_specific_region(a, reg_id, regions, ox, oy)
#     # Operation.
#     opt = 0
#     while True:
#         # get specific region.
#         d = mask_specific_region(res_dil, reg_id, regions, ox, oy)
#         # Duplication.
#         be_d = d.copy()
#         # Dilation the specific region.
#         if opt == 0:
#             for _ in range(1):
#                 # for _ in range(7):
#                 d = morphology.dilation(d, selem=morphology.disk(5))
#             d = mask_specific_region(d, reg_id, regions, ox, oy)
#         # # Erosion the specific region.
#         # if opt == 1:
#         #     # Dilation the opposition is the same function.
#         #     template = np.ones(d.shape, dtype=np.uint8)
#         #     cur_reg = mask_specific_region(template, reg_id, regions, ox, oy)
#         #     opposition = cur_reg - d
#         #     for _ in range(1):
#         #     # for _ in range(7):
#         #         opposition = morphology.dilation(opposition, selem=morphology.disk(5))
#         #     opposition = mask_specific_region(opposition, reg_id, regions, ox, oy)
#         #     d = cur_reg - opposition
#
#         # print('--- id = {}  ---'.format(reg_id))
#         # print(d)
#         # print(org)
#         # print(d-org)
#         # print('---------')
#
#         # Judge whether terminated or not.
#         diff = org - d
#         if (diff[np.where(org != 1)] != 0).any():
#         # if not (diff[np.where(org == 1)] == 0).all():
#             break
#         # Erase the selected region.
#         res_dil -= be_d
#         # Record the operation. (Add the region that previously erased)
#         res_dil += d
# print(time.time() - start)
#
# # Visualization.
# plt.subplot(131)
# plt.imshow(a)
# plt.subplot(132)
# plt.imshow(res)
# plt.subplot(133)
# plt.imshow(res_dil)
# plt.show()
#
#
#
# # start = time.time()
# # # epoch = 7
# # epoch = 20
# # res = b.copy()
# # process = np.zeros((a.shape[0], a.shape[1], epoch))
# # for p in range(epoch):
# #     # get region id.
# #     reg_id = random.randint(0, regions-1)
# #     # get operation.
# #     opt = random.randint(0, 1)
# #     # get specific region.
# #     d = mask_specific_region(res, reg_id, regions, ox, oy)
# #     # Erase the selected region.
# #     res -= d
# #     # Dilation the specific region.
# #     if opt == 0:
# #         for _ in range(2):
# #         # for _ in range(7):
# #             d = morphology.dilation(d, selem=morphology.disk(5))
# #         d = mask_specific_region(d, reg_id, regions, ox, oy)
# #     # Erosion the specific region.
# #     if opt == 1:
# #         # Dilation the opposition is the same function.
# #         template = np.ones(d.shape, dtype=np.uint8)
# #         cur_reg = mask_specific_region(template, reg_id, regions, ox, oy)
# #         opposition = cur_reg - d
# #         for _ in range(2):
# #         # for _ in range(7):
# #             opposition = morphology.dilation(opposition, selem=morphology.disk(5))
# #         opposition = mask_specific_region(opposition, reg_id, regions, ox, oy)
# #         d = cur_reg - opposition
# #     # Record the operation. (Add the region that previously erased)
# #     res += d
# #     # Record the process.
# #     process[:, :, p] = res
# #     # Console output.
# #     print(d)
# #     print(res)
# # print(time.time() - start)
# #
# # for i in range(epoch):
# #     # Visualization.
# #     s = i % 7
# #     win_id = 330 + s + 1
# #     plt.subplot(win_id)
# #     plt.imshow(process[:, :, i], cmap='gray')
# # plt.subplot(338)
# # plt.imshow(b, cmap='gray')
# # plt.subplot(339)
# # plt.imshow(res, cmap='gray')
# # plt.show()
#
#
#
#
#
# print('-----------------  End Here  ---------------------')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
