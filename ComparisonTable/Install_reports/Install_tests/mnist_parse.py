from mnist import MNIST
#you can find python-mnist source code on https://github.com/sorki/python-mnist

datahandler = MNIST('python-mnist/data')
train_data = datahandler.load_training()
for i in range(59989, 60000):
    print('IMAGE: {}'.format(i+1))
    img = train_data[0][i]
    print(datahandler.display(img))
    print('LABEL: {}\n\n\n'.format(train_data[1][i]))
