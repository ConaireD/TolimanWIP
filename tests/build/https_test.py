from toliman.build.https import (
    get_https_stream,
    dowload_byte_from_https,
    dowload_file_from_https,
)

def test_get_https_stream_on_success():
    # Arrange
    stream: iter = get_https_stream("https://github.com")

    # Act/Assert
    assert stream.status_code == 200
    
def test_get_https_stream_on_failure():
    # Arrange/Act/Assert
    with pytest.raises(ValueError):
        stream: iter = get_https_stream("https://i'm/not/a/website.txt")
