import h5py
import numpy as np

#Read the hdf5：
f = h5py.File('D:\pyWorkSpace\Trip1.hdf','r')   #open h5 files
# read all keys from that group


for i in f.keys():
    group = f[i]
    print(i+'/////////////////')
    for key in group.keys():
        print(key)

print(len(f['Plugins']['Body_acceleration_X']))

'''
print(f['AI'])
print(f['CAN'])
print(f['GPS'])
print(f['Math'])
print(f['Plugins'])
'''
#trainset=np.zeros([100000,16])

# for i in range (100000):
#     trainset[i][0] = f['Plugins']['Accelerometer_X'][i][0]
#     trainset[i][1] = f['Plugins']['Body_acceleration_X'][i][0]
#     trainset[i][2] = f['GPS']['Acceleration'][i][0]
#     trainset[i][3] = f['CAN']['AccPedal'][i][0]
#     trainset[i][4] = f['Plugins']['Velocity_X'][i][0]
#     trainset[i][5] = f['CAN']['VehicleSpeed'][i][0]
#     trainset[i][6] = f['CAN']['EngineSpeed_CAN'][i][0]
#     trainset[i][7] = f['CAN']['WheelSpeed_FL'][i][0]
#     trainset[i][8] = f['CAN']['WheelSpeed_FR'][i][0]
#     trainset[i][9] = f['CAN']['WheelSpeed_RL'][i][0]
#     trainset[i][10] = f['CAN']['WheelSpeed_RR'][i][0]
#     trainset[i][11] = f['CAN']['BoostPressure'][i][0]
#     trainset[i][12] = f['CAN']['BrkVoltage'][i][0]
#     trainset[i][13] = f['CAN']['EngineTemperature'][i][0]
#     trainset[i][14] = f['CAN']['SteerAngle1'][i][0]
#     trainset[i][15] = f['CAN']['Yawrate1'][i][0]

# np.savetxt("D:\pyWorkSpace/train.txt",trainset)
# print(trainset)
    
