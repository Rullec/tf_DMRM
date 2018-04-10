# tools: 
import os

record_file = '123'
goal_file = 'new_val.txt'
frame_num = 16

if os.path.exists(goal_file):
	os.remove(goal_file)

def get_num(low, high, num):
	point = []
	high, low, num = int(high), int(low), int(num)
	gap = (high - low + 1 ) * 1.0/ num
	for i in range(0, num):
		tmp = int(i * gap + low)
		tmp = min(high, tmp)
		if tmp<10:
			tmp = '00' + str(tmp)
		elif tmp < 100:
			tmp = '0' + str(tmp)
		else:
			tmp = str(tmp)
		point.append(tmp)
	return point

def main():
	with open(record_file, 'r') as f:
		
		for filestr in f:
			num, name = filestr.split(' ')
			num = int(num)
			name = name[:-1]
			label = int(name[-2:])
			if num<=80:
				point = get_num(1, num, frame_num)
			elif num <=110:
				point = get_num(1+5, num-5, frame_num)
			else:
				point = get_num(num/2-num/4, num/2+num/4, frame_num)

			with open(goal_file, 'a+') as f:
				[f.write(name + '-' + i + '.jpg ' + str(label) + '\n') for i in point]
			print(num, name, label)

print(get_num(1,80, 14))
main()
