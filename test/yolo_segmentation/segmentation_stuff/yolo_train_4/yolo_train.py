from ultralytics import YOLO, checks, hub
checks()

hub.login('71454859f9a2ce11ea390ba04b99fac877ea55f8f7')

model = YOLO('https://hub.ultralytics.com/models/fK6E4iIhOezw1mpM8xd2')
results = model.train()