import numpy as np
import os

def read_intan_data(filename):
        
    # Open the file
    with open(filename, 'rb') as fid:
        # Read first three header bytes encoding file version
        header = np.fromfile(fid, dtype=np.uint8, count=3)
        
        if header[0] != 128:
            raise ValueError('Improper data file format.')
        
        if header[1] != 1 or header[2] != 1:
            print('Data file version may not be compatible with this m-file.')
        
        # Now see which amplifier channels are saved in this file.
        amp_on = np.fromfile(fid, dtype=np.uint8, count=64)   
        
        amps = np.nonzero(amp_on)[0] + 1
        num_amps = len(amps)
            
        # Get the file size
        filesize = os.path.getsize(filename)
        t_count = int((filesize - 67) / (num_amps * 4 + 1))
        t_max = t_count / 25000
        
         
        print('\nData file contains {:.2f} seconds of data from {} amplifier channels.'.format(t_max, num_amps))
        print('Channels:', ' '.join(map(str, amps)))
        
        # Pre-allocate large data matrices.
        t = np.arange(t_count) / 25000
        
        # Go back to the beginning of the file...
        fid.seek(0)
        # ...skip the header this time...
        fid.seek(67)
        
        # Read the entire file
        data2 = np.fromfile(fid, dtype=np.uint8)
        
        # extract the digital data
        aux_data = data2[(num_amps * 4): (filesize - 67): (num_amps * 4 + 1)]
        
        # extract individual bits
        aux = np.unpackbits(aux_data.reshape(-1, 1), axis=1)[:, -6:]
        aux = np.transpose(aux)
        
        # delete the digital data
        delete_indices = np.arange(num_amps * 4, filesize - 67, num_amps * 4 + 1)
        data2 = np.delete(data2, delete_indices)
        
        # convert the remaining data from bytes to single
        data2 = np.frombuffer(data2, dtype=np.float32)
        
        data = data2.reshape((num_amps, -1), order='F')
        
        return t, amps, data, aux



