- add a long-term progression term for multi year optimisation (or a windowed optimisation of alpha/kappa that are similar to the power law to see the evolution of those parameters over time)
- frame this as a data only (without lab) and without access to athletes directly.
- test also a intra race prediction based on the previous segments


# Story to write 
- previous considered that during race athletes at a percentage of their vt2 speed. but it was not considering the intensity intended for the race. it would be inefficient for training race but also for race with much longer distances. we had the %hrr as a moderator and optimised the model using the real hrr. then based on this model we can predict the time for a certain target hrr. the idea is also to  estimate the mean hrr for different duration of race and then try to find the best time a trailer can have based on his unique biological elements
an important part is also to have interpretable parameters that describe physics segment based models and to be able to see the evolution of the parameters over time (overlapping windows or expentially decreasing weighted activities)