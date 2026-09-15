// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Runtime.InteropServices;
using DesireeIA.Native;

namespace DesireeIA;

/// <summary>
/// Wrapper for native image data. Manages the lifecycle of VisionImage
/// allocated by the native engine. Supports images, video frames, and
/// any pixel buffer the model can process.
/// </summary>
public sealed class VisionImageWrapper : IDisposable
{
    private NativeMethods.VisionImage _image;
    private bool _disposed;

    internal VisionImageWrapper(NativeMethods.VisionImage image)
    {
        _image = image;
    }

    /// <summary>Image width in pixels.</summary>
    public uint Width => _image.Width;

    /// <summary>Image height in pixels.</summary>
    public uint Height => _image.Height;

    /// <summary>Number of channels (1=grayscale, 3=RGB, 4=RGBA).</summary>
    public uint Channels => _image.Channels;

    /// <summary>Total pixel count.</summary>
    public uint PixelCount => Width * Height;

    /// <summary>Total byte count of pixel data.</summary>
    public uint ByteCount => PixelCount * Channels;

    /// <summary>Get the native image struct for P/Invoke calls.</summary>
    internal ref NativeMethods.VisionImage GetNative() => ref _image;

    /// <summary>
    /// Copy pixel data to a managed byte array.
    /// </summary>
    public byte[] ToByteArray()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_image.Data == IntPtr.Zero || ByteCount == 0)
            return Array.Empty<byte>();

        var bytes = new byte[ByteCount];
        Marshal.Copy(_image.Data, bytes, 0, (int)ByteCount);
        return bytes;
    }

    /// <summary>
    /// Convert image to base64 string for embedding in chat messages.
    /// </summary>
    public string ToBase64()
    {
        return Convert.ToBase64String(ToByteArray());
    }

    /// <summary>
    /// Get the MIME type based on channel count.
    /// </summary>
    public string MimeType => Channels switch
    {
        1 => "image/grayscale",
        3 => "image/rgb",
        4 => "image/rgba",
        _ => "application/octet-stream"
    };

    /// <summary>
    /// Create an image message for multimodal models.
    /// The image is encoded as base64 with metadata.
    /// </summary>
    public string ToMessageContent(string? textContent = null)
    {
        var b64 = ToBase64();
        var content = $"[image:{Width}x{Height}@{Channels}ch:{b64}]";
        if (!string.IsNullOrEmpty(textContent))
        {
            content += "\n" + textContent;
        }
        return content;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_image.Data != IntPtr.Zero)
        {
            NativeMethods.desireeia_free_image(ref _image);
            _image.Data = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    ~VisionImageWrapper() => Dispose();
}

/// <summary>
/// Represents a stream of data that can be sent to or received from the model.
/// Supports images, video frames, documents, and binary files.
/// </summary>
public sealed class ModelStream
{
    /// <summary>Stream type identifier.</summary>
    public StreamType Type { get; init; }

    /// <summary>MIME type of the stream data.</summary>
    public string MimeType { get; init; } = "";

    /// <summary>Raw stream data.</summary>
    public byte[] Data { get; init; } = Array.Empty<byte>();

    /// <summary>Optional filename for file streams.</summary>
    public string? Filename { get; init; }

    /// <summary>Optional dimensions for image/video streams.</summary>
    public uint? Width { get; init; }
    public uint? Height { get; init; }

    /// <summary>Optional frame count for video streams.</summary>
    public int? FrameCount { get; init; }

    /// <summary>Optional frame rate for video streams.</summary>
    public float? FrameRate { get; init; }

    /// <summary>
    /// Create an image stream from a VisionImageWrapper.
    /// </summary>
    public static ModelStream FromImage(VisionImageWrapper image, string? textContent = null)
    {
        return new ModelStream
        {
            Type = StreamType.Image,
            MimeType = image.MimeType,
            Data = image.ToByteArray(),
            Width = image.Width,
            Height = image.Height
        };
    }

    /// <summary>
    /// Create a file stream from raw bytes.
    /// </summary>
    public static ModelStream FromFile(byte[] data, string filename, string mimeType = "application/octet-stream")
    {
        return new ModelStream
        {
            Type = StreamType.File,
            MimeType = mimeType,
            Data = data,
            Filename = filename
        };
    }

    /// <summary>
    /// Create a video stream from frame data.
    /// </summary>
    public static ModelStream FromVideo(byte[] data, string format = "mp4",
                                         int frameCount = 0, float frameRate = 30f)
    {
        return new ModelStream
        {
            Type = StreamType.Video,
            MimeType = $"video/{format}",
            Data = data,
            FrameCount = frameCount,
            FrameRate = frameRate
        };
    }

    /// <summary>
    /// Create a document stream from text or binary data.
    /// </summary>
    public static ModelStream FromDocument(byte[] data, string filename, string mimeType)
    {
        return new ModelStream
        {
            Type = StreamType.Document,
            MimeType = mimeType,
            Data = data,
            Filename = filename
        };
    }

    /// <summary>
    /// Encode this stream as a message for the model.
    /// </summary>
    public string ToMessageContent()
    {
        var b64 = Convert.ToBase64String(Data);
        return Type switch
        {
            StreamType.Image => $"[image:{Width}x{Height}:{b64}]",
            StreamType.Video => $"[video:{FrameCount}f@{FrameRate}fps:{b64}]",
            StreamType.File => $"[file:{Filename}:{MimeType}:{b64}]",
            StreamType.Document => $"[document:{Filename}:{MimeType}:{b64}]",
            _ => $"[binary:{MimeType}:{b64}]"
        };
    }
}

/// <summary>Types of data streams the model can process or generate.</summary>
public enum StreamType
{
    Image,
    Video,
    File,
    Document,
    Audio
}
