import requests
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO
import subprocess
import sys
import tempfile
import os
import shlex

tesseract_cmd = '/home/arna/Desktop/IITKGP/projects/PilotImage/indicview-base/webApp/rango/tesscv'

def run_tesseract(input_filename, output_filename_base, lang=None, boxes=False, config=None):
    '''
    runs the command:`tesseract_cmd` `input_filename` `output_filename_base`
    
    returns the exit status of tesseract, as well as tesseract's stderr output
    '''
    command = [tesseract_cmd, input_filename, output_filename_base]

    if boxes:
        command += ['batch.nochop', 'makebox']
        
    if config:
        command += shlex.split(config)
    
    print "running tesseract  "
    proc = subprocess.Popen(command,stderr=subprocess.PIPE)
    print "Waiting....."
    status = proc.wait()
    print status
    error_string = proc.stderr.read()
    print error_string
    return (status, error_string)

def cleanup(filename):
    ''' tries to remove the given filename. Ignores non-existent files '''
    try:
        os.remove(filename)
    except OSError:
        pass

def get_errors(error_string):
    '''
    returns all lines in the error_string that start with the string "error"
    '''

    lines = error_string.splitlines()
    error_lines = tuple(line for line in lines if line.find('Error') >= 0)
    if len(error_lines) > 0:
        return '\n'.join(error_lines)
    else:
        return error_string.strip()

def tempnam():
    ''' returns a temporary file-name '''
    tmpfile = tempfile.NamedTemporaryFile(prefix="tess_")
    return tmpfile.name

class TesseractError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)

def image_to_string(image, lang=None, boxes=False, config=None):
    '''
    Runs tesseract on the specified image. First, the image is written to disk,
    and then the tesseract command is run on the image. Resseract's result is
    read, and the temporary files are erased.
    
    also supports boxes and config.
    
    if boxes=True
        "batch.nochop makebox" gets added to the tesseract call
    if config is set, the config gets appended to the command.
        ex: config="-psm 6"
    '''

    if len(image.split()) == 4:
        # In case we have 4 channels, lets discard the Alpha.
        # Kind of a hack, should fix in the future some time.
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    
    input_file_name = '%s.bmp' % tempnam()
    output_file_name_base = '/home/arna/Desktop/IITKGP/projects/PilotImage/indicview-base/results'
    if not boxes:
        output_file_name = '%s.txt' % output_file_name_base
    else:
        output_file_name = '%s.box' % output_file_name_base
    try:
        image.save(input_file_name)
        print "Before Running tess"
        status, error_string = run_tesseract(input_file_name,
        	output_file_name_base,lang=lang,boxes=boxes,config=config)
        print "After Running Tess"
        print "Status: ", status
        print "Errors:   ", error_string
        if status:
            errors = get_errors(error_string)
            raise TesseractError(status, errors)
        f = open(output_file_name_base)
        try:
            return f.read().strip()
        finally:
            f.close()
    finally:
        cleanup(input_file_name)
        cleanup(output_file_name)

def process_image(url):
    image = _get_image(url)
    image.filter(ImageFilter.SHARPEN)
    return image_to_string(image, lang='hin')

def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))
