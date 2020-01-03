from os import rename


""" renaming some pictures in the dataset """


i = 701 ### current picture name
j = 41 #### its new name


suf = ".png"

while i < 1610 :
	pre_j = "-00"
	pre_i = "-0"
	#### create the new and correct name
	if j < 100:
		pre_j += "0"

	new_name = pre_j + str(j) + suf

	if i < 1000:
		pre_i += "0"

	current_name = pre_i + str(i) + suf

	rename(current_name, new_name)

	i+=1
	j+=1

	if (i-41)%100 == 0:
		i += 60

	if (j-1)%100 ==0:
		j += 40

print("rename done.")




