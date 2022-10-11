import mxnet as mx
import time
import gluoncv as gcv
from gluoncv.utils import try_import_cv2

cv2 = try_import_cv2()


net = gcv.model_zoo.get_model(
    # good, fast
    'ssd_512_mobilenet1.0_coco',
    # 'ssd_512_mobilenet1.0_voc',
    # 'ssd_512_mobilenet1.0_voc_int8',
    #
    # 'yolo3_mobilenet1.0_coco',
    # 'yolo3_mobilenet1.0_voc',
    # too slow...
    # 'faster_rcnn_resnet50_v1b_voc',  # too slow...
    # 'faster_rcnn_fpn_syncbn_resnest50_coco',  # too slow...
    pretrained=True)

net.hybridize()


cap = cv2.VideoCapture(0)
time.sleep(1)


while(True):

    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(
        frame, short=512, max_size=700
    )
    # rgb_nd, frame = gcv.data.transforms.presets.yolo.transform_test(
    #     frame, short=512, max_size=700
    # )
    # rgb_nd, frame = gcv.data.transforms.presets.rcnn.transform_test(
    #     frame, short=512, max_size=700
    # )

    class_IDs, scores, bounding_boxes = net(rgb_nd)

    img = gcv.utils.viz.cv_plot_bbox(frame,
                                     bounding_boxes[0],
                                     scores[0],
                                     class_IDs[0],
                                     class_names=net.classes)
    gcv.utils.viz.cv_plot_image(img)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()