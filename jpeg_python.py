import numpy as np
from bitstream import BitStream
import huffmanEncode
import cv2

zigzagOrder = np.array(
[  0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63])

std_luminance_quant_tbl = np.array(
[ 16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99],dtype=int)

std_chrominance_quant_tbl = np.array(
[ 17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99],dtype=int)

dctA = np.zeros(shape=(8, 8))
for i in range(8):
    c = 0
    if i == 0:
        c = np.sqrt(1 / 8)
    else:
        c = np.sqrt(2 / 8)
    for j in range(8):
        dctA[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * 8))

srcFileName = '9.bmp'
outputJPEGFileName = '9.jpeg'
quality = 85

srcImage = cv2.imread(srcFileName)
print(srcImage.shape)
srcImageHeight, srcImageWidth = srcImage.shape[:2]

imageHeight, imageWidth = srcImageHeight, srcImageWidth
if (srcImageHeight % 8 != 0):
    imageHeight = srcImageHeight // 8 * 8 + 8
if (srcImageWidth % 8 != 0):
    imageWidth = srcImageWidth // 8 * 8 + 8
padImage = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
padImage[:srcImageHeight,:srcImageWidth] = srcImage


yImage =   0.299 * padImage[:,:,2] +  0.587 * padImage[:,:,1] + 0.114 * padImage[:,:,0]
uImage = -0.1687 * padImage[:,:,2] - 0.3313 * padImage[:,:,1] +   0.5 * padImage[:,:,0] + 128
vImage =     0.5 * padImage[:,:,2] -  0.419 * padImage[:,:,1] - 0.081 * padImage[:,:,0] + 128

yImage = yImage.astype(np.int32) - 127
uImage = uImage.astype(np.int32) - 127
vImage = vImage.astype(np.int32) - 127

quality = np.clip(quality,1,100)
if(quality < 50):
    qualityScale = 5000 / quality
else:
    qualityScale = 200 - quality * 2

luminanceQuantTbl = np.array(np.floor((std_luminance_quant_tbl * qualityScale + 50) / 100))
luminanceQuantTbl = np.where(luminanceQuantTbl == 0, 1, luminanceQuantTbl)
luminanceQuantTbl = np.where(luminanceQuantTbl > 255, 255, luminanceQuantTbl)
luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8]).astype(int)

chrominanceQuantTbl = np.array(np.floor((std_chrominance_quant_tbl * qualityScale + 50) / 100))
chrominanceQuantTbl = np.where(chrominanceQuantTbl == 0, 1, chrominanceQuantTbl)
chrominanceQuantTbl = np.where(chrominanceQuantTbl > 255, 255, chrominanceQuantTbl)
chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)

blockSum = imageWidth // 8 * imageHeight // 8

yDC = np.zeros([blockSum], dtype=int)
uDC = np.zeros([blockSum], dtype=int)
vDC = np.zeros([blockSum], dtype=int)

dyDC = np.zeros([blockSum], dtype=int)
duDC = np.zeros([blockSum], dtype=int)
dvDC = np.zeros([blockSum], dtype=int)

blockNum = 0
sosBitStream = BitStream()
for y in range(0, imageHeight, 8):
    for x in range(0, imageWidth, 8):
        # 对yuv的三个block做DCT
        yDct = np.dot(np.dot(dctA, yImage[y:y + 8, x:x + 8]), dctA.T)
        uDct = np.dot(np.dot(dctA, uImage[y:y + 8, x:x + 8]), dctA.T)
        vDct = np.dot(np.dot(dctA, vImage[y:y + 8, x:x + 8]), dctA.T)
        if y==0 and x==0:
        #     print('Y分量为：',yDct)
            print('u分量为：',uDct)
        #     print('v分量为：',vDct)
        # 量化表量化
        yQuant = np.rint(yDct / luminanceQuantTbl).flatten()
        uQuant = np.rint(uDct / chrominanceQuantTbl).flatten()
        vQuant = np.rint(vDct / chrominanceQuantTbl).flatten()
        # zigzag重排
        yZCode = np.array([yQuant[zigzagOrder[i]] for i in range(64)]).astype(np.int32)
        uZCode = np.array([uQuant[zigzagOrder[i]] for i in range(64)]).astype(np.int32)
        vZCode = np.array([vQuant[zigzagOrder[i]] for i in range(64)]).astype(np.int32)
        # block的DC数据处理
        yDC[blockNum] = yZCode[0]
        uDC[blockNum] = uZCode[0]
        vDC[blockNum] = vZCode[0]
        if(blockNum==0):
            dyDC[blockNum] = yDC[blockNum]
            duDC[blockNum] = uDC[blockNum]
            dvDC[blockNum] = vDC[blockNum]
        else:
            dyDC[blockNum] = yDC[blockNum] - yDC[blockNum-1]
            duDC[blockNum] = uDC[blockNum] - uDC[blockNum-1]
            dvDC[blockNum] = vDC[blockNum] - vDC[blockNum-1]
        # if duDC[blockNum]==0:
            # print(huffmanEncode.encodeDCToBoolList(duDC[blockNum],0, 0))
        # 编码yDC和yAC
        sosBitStream.write(huffmanEncode.encodeDCToBoolList(dyDC[blockNum],1, 0),bool)
        huffmanEncode.encodeACBlock(sosBitStream, yZCode[1:], 1, 0)
        # 编码uDC和uAC
        sosBitStream.write(huffmanEncode.encodeDCToBoolList(duDC[blockNum],0, 0),bool)
        huffmanEncode.encodeACBlock(sosBitStream, uZCode[1:], 0, 0)
        # 编码vDC和uAC
        sosBitStream.write(huffmanEncode.encodeDCToBoolList(dvDC[blockNum],0, 0),bool)
        huffmanEncode.encodeACBlock(sosBitStream, vZCode[1:], 0, 0)
        blockNum = blockNum + 1

jpegFile = open(outputJPEGFileName, 'wb+')
print('打开lena01文件')
# 写入jpeg头
jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))
# 写入y量化表
jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
luminanceQuantTbl = luminanceQuantTbl.reshape([64])
jpegFile.write(bytes(luminanceQuantTbl.tolist()))

print('Y量化表',bytes(luminanceQuantTbl.tolist()))
# 写入c量化表
jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
chrominanceQuantTbl = chrominanceQuantTbl.reshape([64])
jpegFile.write(bytes(chrominanceQuantTbl.tolist()))
# 写入高度和宽度
jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
hHex = hex(srcImageHeight)[2:].rjust(4,'0')
wHex = hex(srcImageWidth)[2:].rjust(4,'0')
jpegFile.write(huffmanEncode.hexToBytes(hHex))
jpegFile.write(huffmanEncode.hexToBytes(wHex))

# 写入子图
jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))
# 写入huffman表
jpegFile.write(huffmanEncode.hexToBytes('FFC401A2000000070101010101000000000000000004050302\
0601000708090A0B0100020203010101010100000000000000010002030405060708090A0B10000201030302040\
20607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F\
12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D\
4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E\
8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0\
100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20\
773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C\
5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F83949596979899\
9A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))

# 这个对jpeg的头解释的很详细了：https://blog.csdn.net/ymlbright/article/details/44179891

# sos扫描数据，确保是8的倍数
sosLength = sosBitStream.__len__()
filledNum = 8 - sosLength % 8
if(filledNum!=0):
    sosBitStream.write(np.ones([filledNum]).tolist(),bool)
jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0])) # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00

# 写入编码数据
sosBytes = sosBitStream.read(bytes)
for i in range(len(sosBytes)):
    jpegFile.write(bytes([sosBytes[i]]))
    if(sosBytes[i]==255):
        jpegFile.write(bytes([0])) # FF to FF 00


# 写入结束符
jpegFile.write(bytes([255,217])) # FF D9
jpegFile.close()

