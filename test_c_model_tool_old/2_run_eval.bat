@echo off
setlocal enabledelayedexpansion

set TXT_DIR=preprocessed_layer5_txt
set OUTPUT_DIR=c_outputs

:: Tạo thư mục nếu chưa có
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ========================================
echo BAT DAU CHAY C MODEL TREN WINDOWS...
echo ========================================

set count=0
for %%f in ("%TXT_DIR%\*.txt") do (
    set /a count+=1
    
    :: Lấy tên file (không lấy đuôi .txt và đường dẫn)
    set "filename=%%~nf"
    
    :: Chạy file .exe và lưu kết quả
    :: Chú ý: File C của bạn trên Windows khi build sẽ có đuôi .exe
    a.exe "%%f" 
    
    :: In log mỗi 50 ảnh
    set /a mod=!count!%%50
    if !mod! == 0 (
        echo   -^> Da chay qua C: !count! anh
    )
)

echo [V] Hoan thanh! Ket qua da luu vao thu muc: %OUTPUT_DIR%\
pause