from preprocess import preprocesses

# input_datadir = "C:\Users\anja.kovacevic\train_img."
# output_datadir = "C:\Users\anja.kovacevic\process_anja_face."
# input_datadir = 'C:/Users/anja.kovacevic/train_img'
# output_datadir = 'C:/Users/anja.kovacevic/process_anja_face'
# input_datadir = 'C:\\Users\\anja.kovacevic\\train_img'
# output_datadir = 'C:\\Users\\anja.kovacevic\\process_anja_face'
# input_datadir = r'C:\Users\anja.kovacevic\train_img\Anja'
# output_datadir = r'C:\Users\anja.kovacevic\process_anja_face'
input_datadir = './train_img'
output_datadir = './process_anja_face'


obj = preprocesses(input_datadir, output_datadir)
nrof_images_total, nrof_successfully_aligned = obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


