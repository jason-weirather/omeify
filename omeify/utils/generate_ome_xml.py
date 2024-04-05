from lxml import etree


import os
import omeify
import uuid
import xmltodict

def get_map_annotation(xml_path):
    tree = etree.parse(xml_path)
    map_annotation = tree.find(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}MapAnnotation")
    return map_annotation

def generate_ome_xml(tiff_features,zarr_object,display_uuid=True, rename_channels = {}):
    myuuid = None
    if display_uuid: myuuid = str(uuid.uuid4())
    # Create the root element with the specified namespace and attributes
    namespaces = {
        None: "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
     }
    root = etree.Element("OME", {
        etree.QName(namespaces["xsi"], "schemaLocation"): "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
    }, nsmap=namespaces)

    # Add optional attributes to the root element
    if display_uuid: root.set("UUID", f"urn:uuid:{myuuid}")
    #root.set("Creator", f'omeify v{omeify.__version__}')


    # Create an Image element
    image = etree.SubElement(root, "Image", 
        ID=f"{tiff_features.image_id}", 
        Name=f"{tiff_features.name}")

    # Create the Pixels element and add it to the Image
    pixels = etree.SubElement(
        image, "Pixels", 
        BigEndian=f"{tiff_features.big_endian}", 
        DimensionOrder=f"{tiff_features.dimension_order}", 
        ID=f"{tiff_features.pixel_id}", 
        Interleaved=f"{tiff_features.interleaved}",
        PhysicalSizeX=f"{tiff_features.physical_size_x}", 
        PhysicalSizeXUnit=f"{tiff_features.physical_size_x_unit}", 
        PhysicalSizeY=f"{tiff_features.physical_size_y}",
        PhysicalSizeYUnit=f"{tiff_features.physical_size_y_unit}", 
        SignificantBits=f"{tiff_features.significant_bits}", 
        SizeC=f"{tiff_features.size_c}", 
        SizeT=f"{tiff_features.size_t}", 
        SizeX=f"{tiff_features.size_x}", 
        SizeY=f"{tiff_features.size_y}",
        SizeZ=f"{tiff_features.size_z}",
        Type=f"{tiff_features.type}")

    # Create the Channel elements and add them to the Pixels
    for c in tiff_features.channels:
        channel = etree.SubElement(pixels, "Channel", 
            ID=f"{c['ID']}", 
            Name=f"{c['Name'] if c['Name'] not in rename_channels else rename_channels[c['Name']]}", 
            SamplesPerPixel=f"{c['SamplesPerPixel']}")
        etree.SubElement(channel, "LightPath")
    
    #etree.SubElement(pixels, "MetadataOnly")

    # Create a MapAnnotation element and add it to the StructuredAnnotations
    etree.SubElement(pixels, "TiffData", IFD="0", PlaneCount=f"{tiff_features.plane_count}")

    # Get the build OME tiff

    map_annotation = get_map_annotation(os.path.join(zarr_object.store.path,'OME','METADATA.ome.xml'))

    # Create a StructuredAnnotations element
    structured_annotations = etree.SubElement(root, "StructuredAnnotations")

    if map_annotation is not None:
        # Smaller images may not have map annotation
        structured_annotations.append(map_annotation)

    # To view the XML tree as a string, use the following code:
    xml_string = etree.tostring(root, pretty_print=False, encoding="utf-8", xml_declaration=True).decode("utf-8")
    
    return {'xml_string':xml_string, 'uuid':myuuid}
