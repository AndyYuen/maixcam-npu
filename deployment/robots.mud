[basic]
type = cvimodel
model = robots_cv181x_int8_sym.cvimodel

[extra]
model_type = yolov5
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
anchors=19.0, 27.0, 44.0, 40.0, 38.0, 94.0, 96.0, 68.0, 86.0, 152.0, 180.0, 137.0, 140.0, 301.0, 303.0, 264.0, 238.0, 542.0, 436.0, 615.0, 739.0, 380.0, 925.0, 792.0
labels = B-Robot, R-Robot
