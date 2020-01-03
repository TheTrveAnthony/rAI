from os import rename


""" renaming some pictures in the dataset (for validation set) """


i = 1611 ### current picture name
j = 0 #### its new name


suf = ".png"
pre_i = "-0"
while i < 1831 :
	pre_j = "-000"
	
	#### create the new and correct name
	if j < 10:
		pre_j += "0"

	new_name = pre_j + str(j) + suf


	current_name = pre_i + str(i) + suf

	rename(current_name, new_name)

	i+=1
	j+=1

	if (i-41)%100 == 0:
		i += 60


print("rename done.")




