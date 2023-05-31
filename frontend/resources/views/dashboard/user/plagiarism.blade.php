@extends('dashboard.layouts.master')
@section('css')
<link href="https://unpkg.com/filepond/dist/filepond.css" rel="stylesheet">
<link href="https://unpkg.com/filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css" rel="stylesheet">
<link rel="stylesheet" href="{{ asset('assets/vendors/toastify/toastify.css') }}">
@endsection

@section('content')
<div class="page-heading">
    <h3>{{ $titlePage }}</h3>
</div>
<div class="page-content">
    <input type="text" value="{{ Auth::user()->id}}" id="user_id" hidden>
    <div class="card">
        <div class="card-header">
            <h5 class="card-title">Upload File</h5>
        </div>
        <div class="card-content">
            <div class="card-body">
                <!-- Basic file uploader -->
                <input type="file" class="key-answer">
                <button type="submit" class="btn btn-info" id='btn-keyAnswer'>Submit Form</button>
            </div>
        </div>
    </div>
</div>
<div class="table-responsive">
    <table id="data-score" class="table table-striped">
        <thead>
            <tr>
                <td>Nama</td>
                <td>Kelas</td>
                <td>No Absen</td>
                <td>Nilai Plagiarism 1</td>
                <td>Nilai Plagiarism 2</td>
                <td>Nilai Plagiarism 3</td>
                <td>Nilai Plagiarism 4</td>
                <td>Nilai Plagiarism 5</td>
                <td>Total Plagiarism</td>
            </tr>
        </thead>
    </table>
</div>
@endsection

@section('js')
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"
    integrity="sha512-894YE6QWD5I59HgZOGReFYm4dnWc1Qt5NtvYSaNcOP+u1T9qYdvdihz0PPSiiqn/+/3e7Jo4EaG7TubfWGUrMQ=="
    crossorigin="anonymous" referrerpolicy="no-referrer"></script>

{{-- toastify --}}
<script src="{{ asset('assets/vendors/toastify/toastify.js') }}"></script>
{{-- <script src="{{ asset('assets/js/extensions/toastify.js') }}"></script> --}}

<!-- filepond validation -->
<script src="https://unpkg.com/filepond-plugin-file-validate-size/dist/filepond-plugin-file-validate-size.js"></script>
<script src="https://unpkg.com/filepond-plugin-file-validate-type/dist/filepond-plugin-file-validate-type.js"></script>

<!-- image editor -->
<script src="https://unpkg.com/filepond-plugin-image-exif-orientation/dist/filepond-plugin-image-exif-orientation.js">
</script>
<script src="https://unpkg.com/filepond-plugin-image-crop/dist/filepond-plugin-image-crop.js"></script>
<script src="https://unpkg.com/filepond-plugin-image-filter/dist/filepond-plugin-image-filter.js"></script>
<script src="https://unpkg.com/filepond-plugin-image-preview/dist/filepond-plugin-image-preview.js"></script>
<script src="https://unpkg.com/filepond-plugin-image-resize/dist/filepond-plugin-image-resize.js"></script>
<script src="https://unpkg.com/filepond/dist/filepond.js"></script>
<script>
    // register desired plugins...
	FilePond.registerPlugin(
        // validates the size of the file...
        FilePondPluginFileValidateSize,
        // validates the file type...
        FilePondPluginFileValidateType,

        // calculates & dds cropping info based on the input image dimensions and the set crop ratio...
        FilePondPluginImageCrop,
        // preview the image file type...
        FilePondPluginImagePreview,
        // filter the image file
        FilePondPluginImageFilter,
        // corrects mobile image orientation...
        FilePondPluginImageExifOrientation,
        // calculates & adds resize information...
        FilePondPluginImageResize,
    );

    // Filepond: Basic
    let keyAnswer = FilePond.create( document.querySelector('.key-answer'));

    let studentsAnswer = FilePond.create( document.querySelector('.students-answer'), {
        allowImagePreview: false,
        allowMultiple: false,
        allowFileEncode: false,
        required: false
    });

    $('#btn-keyAnswer').click(()=> {
        $('#btn-keyAnswer').hide()
        keyAnswer = keyAnswer.getFile().file;
        user_id = $('#user_id').val();
        var formData = new FormData();
        formData.append("_token", "{{ csrf_token() }}");
        formData.append("file", keyAnswer);
        formData.append('user_id', user_id);
        // set ajax
        $.ajax({
            url : 'http://127.0.0.1:5000/check-plagiarism',
            type: 'POST',
            contentType: false,
            processData: false,
            data: formData
        }).then((res)=> {
            Toastify({
                text: "Success Created Model",
                duration: 3000,
                close:true,
                gravity:"top",
                position: "center",
                backgroundColor: "#4fbe87",
            }).showToast();
            $.each(res.data, function(key, val){
                let nilai1 = parseInt(val.plagiarism1);
                let nilai2 = parseInt(val.plagiarism2);
                let nilai3 = parseInt(val.plagiarism3);
                let nilai4 = parseInt(val.plagiarism4);
                let nilai5 = parseInt(val.plagiarism5);
                let totalNilai = Math.round((nilai1+nilai2+nilai3+nilai4+nilai5) / 5)
                $('#data-score').append(
                    "<tbody>"
                        +"<tr>"
                            +"<td>"+val.nama+"</td>"
                            +"<td>"+val.kelas+"</td>"
                            +"<td>"+val.absen+"</td>"
                            +"<td>"+val.plagiarism1+"</td>"
                            +"<td>"+val.plagiarism2+"</td>"
                            +"<td>"+val.plagiarism3+"</td>"
                            +"<td>"+val.plagiarism4+"</td>"
                            +"<td>"+val.plagiarism5+"</td>"
                            +"<td>"+ totalNilai +"</td>"
                        +"</tr>"
                    +"</tbody>"
                )
            })

            // export to excel
            var downloadLink
            var dataType = "application/vnd.ms-excel"
            var tableSelect = document.getElementById('data-score')
            var tableHTML = tableSelect.outerHTML.replace(/ /g, '%20')

            fileName = 'Plagiarism_Report.xls'

            downloadLink = document.createElement('a')

            document.body.appendChild(downloadLink)

            if(navigator.msSaveOrOpenBlob){
                var blob = new Blob(['\ufeff', tableHTML], {
                    type: dataType
                })
                navigator.msSaveOrOpenBlob(blob, fileName)
            } else {
                downloadLink.href = 'data:' + dataType + ',' + tableHTML

                downloadLink.download = fileName

                downloadLink.click()
            }
        }).catch((res)=> {
            console.log(res)
        });
    });
</script>
@endsection
