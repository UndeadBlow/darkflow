from lxml import etree

def add_to_element(el, box):
   rect = etree.SubElement(el, 'rect')
   w, h = box.w, box.h
   x, y = box.x, box.y
   rect.text = '{} {} {} {}'.format(x, y, w, h)
   # dist = etree.SubElement(el, 'distance')
   # dist.text = '0.'
   t = etree.SubElement(el, 'type')
   t.text = str(box.class_num + 1)
   t = etree.SubElement(el, 'subtype')
   t.text = '0'
   comm = etree.SubElement(el, 'comment')
   comm.text = '""'
   pass

def points_to_xml(boxes, classes, skipped = False):
   # of one frame
   ocv_storage = etree.Element('opencv_storage')
   xml = etree.ElementTree(ocv_storage)
   skip = etree.SubElement(ocv_storage
   , 'skipped')
   skip.text = "1" if skipped else "0"
   objs = etree.SubElement(ocv_storage
   , 'Objects')

   added = 0
   for box in boxes:
       if box.class_num in classes:
           el = etree.SubElement(objs, '_')
           add_to_element(el, box)
           added += 1

   if added < 1:
       return ''

   xml_str = '{}'.format(etree.tostring(xml, encoding='utf8', method='xml'))
   # I don't know why b' at the start of file comes here but it comes. Little hotfix
   if (xml_str[: 2] == 'b\''):
       xml_str = xml_str[2 : ]
   if (xml_str[-1 : ] == '\''):
       xml_str = xml_str[: -1]

   return '<?xml version="1.0"?>\n' + xml_str

#
# def el_from_xml(elxml, cname):
#    coords = {'ix' : -1, 'iy' : -1, 'x' : -1, 'y' : -1}
#    el = {
#       'coords'   : coords,
#       'distance' : None,
#       'class'    : 'C1'
#    }
#    for ch in elxml.getchildren():
#       if ch.tag == 'rect':
#          coords['ix'], coords['iy'], w, h = [int(v) for v in ch.text.split()]
#          coords['x'] = coords['ix']+ w
#          coords['y'] = coords['iy']+ h
#       elif ch.tag == 'distance':
#          d = float(ch.text)
#          el['distance'] = d if d > 0 else None
#       elif ch.tag == 'type':
#          el['class'] = '{}{}'.format(cname, ch.text)
#       elif ch.tag == 'comment':
#          pass
#    return el
#
# def xml_to_points(xmlfname):
#    points = []
#    skipped = False
#    tree = etree.parse(xmlfname)
#    for child in tree.getroot().getchildren():
#       if child.tag == 'skipped':
#          skipped = child.text == '1'
#       if child.tag not in xml_names_to_classes:
#          continue
#       cname = xml_names_to_classes[child.tag]
#       points.extend(el_from_xml(el, cname) for el in child.getchildren())
#    return points, skipped
