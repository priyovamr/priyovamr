f = open('Autonomous/tes_single_0.txt')
lines = f.readlines()
latd=[]
lotd = []
for x in lines:
    latd.append(x.split(' ')[4] + " " + x.split(' ')[5] + " " + x.split(' ')[6])
    lotd.append(x.split(' ')[8] + " " + x.split(' ')[9] + " " + x.split(' ')[10])

latd.pop(0)
lotd.pop(0)
print(latd[0])