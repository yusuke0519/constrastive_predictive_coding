# # -*- coding: utf-8 -*-
from fastprogress import master_bar, progress_bar
from time import sleep

mb = master_bar(range(10))
for i in mb:
    for j in progress_bar(range(100), parent=mb):
        sleep(0.01)
        mb.child.comment = 'second bar stat'
    mb.first_bar.comment = 'first bar stat'
    mb.write('Test')
