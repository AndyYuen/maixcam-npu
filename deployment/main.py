from maix import camera, display, image, nn, app, time

# set up yolo5 model
detector = nn.YOLOv5(model="/root/models/robots.mud", dual_buff = True)

# set up camera and display
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
dis = display.Display()

# set up frame rate inference execution time counters
then = time.time_ms()
count = 0
fps = 0
execTime = 0
avgTime = 0

# image processing loop
while not app.need_exit():
    img = cam.read()
    # time detect robots call with a confidence threshold of 0.7
    start = time.time_ms()
    objs = detector.detect(img, conf_th = 0.7)
    execTime = execTime + time.time_ms() - start
    for obj in objs:
        # draw enclosing rectangle and label in the color of the robot
        color = image.COLOR_RED
        if detector.labels[obj.class_id] == 'B-Robot':
            color = image.COLOR_BLUE
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color, 0.8)

    # calculate/display frame rate and average inference execution time  
    count = count + 1
    now = time.time_ms()
    elapsed = now - then
    if elapsed > 1000:
        fps = round(float(count) * 1000. / float(elapsed))
        avgTime = round(float(execTime) / float(count))
        execTime = 0
        then = now
        count = 0
          
    img.draw_string(10, 10, f"{fps} FPS, {avgTime} ms inference", image.COLOR_GREEN, 0.8)

    dis.show(img)
