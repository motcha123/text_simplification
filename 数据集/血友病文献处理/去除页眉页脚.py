import pypdf


def remove_header_from_pdf(input_pdf_path, output_pdf_path):
    # 打开输入的 PDF 文件
    with open(input_pdf_path, 'rb') as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        pdf_writer = pypdf.PdfWriter()

        # 遍历每一页
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            # 从每一页的顶部移除前 100 个单位（假设为页眉的高度）
            page.mediabox.upper_right = (page.mediabox.upper_right[0], page.mediabox.upper_right[1] - 100)
            pdf_writer.add_page(page)

        # 将修改后的 PDF 写入输出文件
        with open(output_pdf_path, 'wb') as output_pdf_file:
            pdf_writer.write(output_pdf_file)

# 调用函数以从输入的 PDF 中删除页眉，并将其保存为 output.pdf
remove_header_from_pdf("test.pdf", "output.pdf")

# 打印成功消息
print("已从 input.pdf 中删除页眉，并保存为 output.pdf")