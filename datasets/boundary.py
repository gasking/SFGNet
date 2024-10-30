from PIL import Image
import cv2
import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    box_w, box_h  = det_size
    a1 = 1
    b1 = (box_w + box_h)
    c1 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (box_w + box_h)
    c2 = (1 - min_overlap) * box_w * box_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_w + box_h)
    c3 = (min_overlap - 1) * box_w * box_h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def draw_gassuian(heat,x,y,radius,feature_w,feature_h):
    sigma = radius
    for j in range(y-3 * int(radius) - 1,y+3 * int(radius) + 1):
     for i in range(x-3 * int(radius) - 1,x+3 * int(radius) + 1):
      if (i < feature_w) and j < (feature_h):
         heat[j,i] = np.exp(- (i - x)**2 / (2*sigma**2) - (j - y)**2 / (2*sigma**2))
    return heat



def convert_boundary(image,stride,vis = False):

    if isinstance(image,Image.Image):
        image = np.array(image)

    h,w = np.array(image).shape

    heat_map = np.zeros((h//stride,w//stride,1))

    #image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    heri ,_ = cv2.threshold(image,0,200,cv2.THRESH_BINARY)

    ksize = np.array([-1,1,1,-1,8,-1,-1,1,1]).reshape((-1,3))
    lap = cv2.Laplacian(_,cv2.CV_8U,ksize)
    canny = cv2.Canny(_, 0, 180)
    lap = cv2.dilate(lap,kernel = (2,2))

    lap = cv2.resize(lap,(h//stride,w//stride))

    boundary = np.zeros((h//stride,w//stride,1))

    boundary[lap > 0.1] = 1

    judge,count = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 坐标点
    boxes = []

    for box in judge:
        boxes.extend([cv2.boundingRect(np.array(box))])

    boxes = np.array(boxes)

    feature_h,feature_w,c = heat_map.shape

    if stride in [2,4]:
        for box in boxes:
            x1,y1 = box[0],box[1]
            x2,y2 = x1 + box[2],y1 + box[3]

            if vis:
             cv2.rectangle(image, (x1, y1), (x2, y2),
                          (255, 0, 0), 2, cv2.LINE_AA)

            center_x = (x1 + x2) // 2 // stride
            center_y = (y1 + y2) // 2 // stride
            w = box[2] // stride
            h = box[3] // stride

            # TODO: reference
            radius =  0.5 * gaussian_radius((w, h))  / 3
            draw_gassuian(heat_map,center_x,center_y,radius,feature_w,feature_h)

    if vis:
        cv2.imshow(f'im_{stride}', image)
        cv2.imshow(f'heat_{stride}', heat_map *255)
        cv2.imshow(f'boundary_{stride}',boundary * 255)
        cv2.imshow(f'lap_{stride}',lap)
        cv2.waitKey(0)

    return heat_map,boundary

if __name__ == '__main__':

    path = r'asset\1011.tif'

    # image = np.array(Image.open(path))
    #
    # hx2,bx2 = convert_boundary(image,2)
    #
    # hx4,bx4 = convert_boundary(image,16)
    #
    # print(hx2.shape,bx2.shape)
    # print(hx4.shape,bx4.shape)



