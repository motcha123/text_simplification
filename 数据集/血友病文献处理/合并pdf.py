import pypdf
import fitz


if __name__ == '__main__':
    # 将图片转换为pdf
    # image_to_pdf = fitz.open()
    # image_path = r'D:\开题报告最终材料\2.jpg'  # 要转换为pdf的图片
    # image_file = fitz.open(image_path)
    # pdf_image = image_file.convert_to_pdf()
    # pdf_image = fitz.open(r'D:\开题报告最终材料\2.pdf', pdf_image)
    # image_to_pdf.insert_pdf(pdf_image)
    # image_to_pdf.save(r'D:\开题报告最终材料\2.pdf')  # 转换后pdf的保存路径

    # 调整一下转换为pdf后的图片的大小
    # reader = pypdf.PdfReader(r'D:\开题报告最终材料\提交给系统\2.pdf')  # 调整大小前的pdf
    # page = reader._get_page(0)
    # print(page.mediabox)
    # page.scale_to(595.32, 841.92)
    # print(page.mediabox)
    # writer = pypdf.PdfWriter()
    # pdf_new = open(r'D:\开题报告最终材料\提交给系统\5.pdf', 'wb')  # 调整大小后的pdf
    # writer.add_page(page)
    # writer.write(pdf_new)

    # 合并多个pdf
    pdf_1 = open(r'D:\qq_data\632822424\FileRecv\MobileFile\1.pdf', 'rb')
    pdf_2 = open(r'D:\qq_data\632822424\FileRecv\MobileFile\2.pdf', 'rb')
    pdf_3 = open(r'D:\qq_data\632822424\FileRecv\MobileFile\3.pdf', 'rb')
    pdf_4 = open(r'D:\qq_data\632822424\FileRecv\MobileFile\4.pdf', 'rb')
    pdf_merge = open(r'D:\qq_data\632822424\FileRecv\MobileFile\merger.pdf', 'wb')
    pdf_writer = pypdf.PdfWriter()
    pdf_writer.merge(0, pdf_1, pages=(0, 1, 1))
    pdf_writer.merge(1, pdf_2)  # 在第0页之前插入pdf_2
    pdf_writer.merge(2, pdf_3)
    pdf_writer.merge(3, pdf_4)
    pdf_writer.write(pdf_merge)
    pdf_merge.close()
