import platform


class Device:
    def __init__(self, serial_number=None):
        '''Abstraction of HID device so that the interface is the same across
        platforms

        Raises
        ------
        TypeError
            If operating system is not Darwin, Linux or Windows

        Attributes
        ----------
            device : obj
                hid device
            type : int
                indicates how to interact with hid device. if 1, send message as
                bytes and provide size argument to read; if 0, send message as
                bytearray without leading 0
        '''
        if platform.system() == 'Linux':
            import easyhid
            self.exception = easyhid.easyhid.HIDException
            en = easyhid.Enumeration()
            devices = en.find(vid=1155, pid=22352, serial=serial_number)
            if len(devices) == 0:
                print('[ERROR] No device found. Ensure that the xarm is connected via '
                      'usb cable and the power is on.\n'
                      'Turn the xarm off and back on again if needed.')
                exit()
            elif len(devices) > 1:
                serial_numbers = ', '.join([f"\t{d.serial_number}" for d in devices])
                print('[ERROR] More than 1 xarm device found with the following serial numbers: \n'
                      f'    {serial_numbers}\n'
                      '  You must specify the serial number in this case.')
                exit()
            else:
                self.device = devices[0]
                self.serial_number = self.device.serial_number
                self.device.open()
                self.type = 0
        elif platform.system() == 'Windows':
            import hid
            self.device = hid.Device(vid=1155, pid=22352, serial=serial_number)
            self.type = 1
        elif platform.system() == 'Darwin':
            import hid
            self.device = hid.device()
            self.device.open(1155, 22352, serial_number)
            self.serial_number = self.device.serial
            self.type = 0
        else:
            raise TypeError('unsupported operating system')

    def write(self, msg):
        '''Write message to device

        Parameters
        ----------
        msg : list
            list of data to be written
        '''
        try:
            if self.type == 0:
                self.device.write(bytearray(msg[1:]))
            elif self.type == 1:
                self.device.write(bytes(msg))
        except self.exception:
            pass

    def read(self, timeout):
        '''Read message from device

        Parameters
        ----------
        timeout : int
            timeout period in milliseconds

        Returns
        -------
        array_like
            message received from device
        '''
        if self.type == 0:
            return self.device.read(timeout)
        elif self.type == 1:
            size = 32
            return self.device.read(size, timeout)

    def close(self):
        '''Close hid device if possible
        '''
        if self.type == 0:
            self.device.close()
