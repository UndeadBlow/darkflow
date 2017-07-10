import os
import serialize as serial
import slicer
import box as BB
import operator
import copy

def GetAllFilesList(path):
    files = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
    return filter(lambda file: 'directory' not in file, files)

def GetAllFilesListRecusive(path, extensions):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all

if __name__ == "__main__":

    path = '/media/undead/Work/Datasets/SignsYOLODataset'
    txt_files = GetAllFilesListRecusive(path, ['_o.xml'])

    for file in txt_files:
        with open(file, 'r') as f:
            data = f.read()

        if len(data) < 10:
            print('removing ', file)
            os.remove(file)


    # for file in txt_files:
    #     print('Processing ', file)
    #     with open(file, 'r') as f:
    #         data = f.read()
    #
    #     if not data:
    #         continue
    #
    #     xml_filename = file.replace('.txt', '_o.xml')
    #
    #     lines = data.split('\r')
    #     signs = []
    #
    #     box = BB.BoundBox(0)
    #     boxes = []
    #     for line in lines:
    #         if not line:
    #             continue
    #         temp = line.split('from')
    #         if len(temp) < 2:
    #             continue
    #         size = temp[0].split('x')
    #         coords = temp[1].split(',')
    #
    #         box.w = int(size[0][size[0].find('[') + 1: ])
    #         box.h = int(size[1])
    #
    #         box.x = int((coords[0][coords[0].find('(') + 1 : ]))
    #         box.y = int((coords[1][ : coords[1].find(')')]))
    #
    #         print('Got BoundBox ', box.x, box.y, box.w, box.h)
    #
    #         boxes.append(copy.deepcopy(box))
    #
    #     xml = serial.points_to_xml(boxes, [0])
    #
    #     with open(xml_filename, 'w') as f:
    #         f.write(xml)
