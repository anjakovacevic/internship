import caffe

model = open('itracker_iter_92000.caffemodel', 'rb')
binary_content = model.read()

protobuf = caffe.proto.caffe_pb2.NetParameter()
protobuf.ParseFromString(binary_content)

layers = protobuf.layers

# now see if there are any layers
print(len(layers)) # outputs 0