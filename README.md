# Visual-computing-find-horizon-of-a-graph
Use opencv framework to detect the horizon in a picture
</br> step1: Use canny edge detector to find all the lines in the picure
</br> step2: Filter out short lines
</br> step3: Filter out vertical lines
</br> step4: Perform polynomial regression on the remaining lines (using the coordinator of the points on a line)
</br> Then we get our detected/predicted horizon line.
Result can be seen in the .jpg files
