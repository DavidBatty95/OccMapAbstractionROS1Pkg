// Auto-generated. Do not edit!

// (in-package gazebo_radiation_plugins.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class GenRandomEnvironmentalEffectsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.folder = null;
      this.number_of_radiation_sources = null;
      this.number_of_rusting_models = null;
      this.number_of_bending_models = null;
      this.max_bend_factor = null;
      this.max_rust_factor = null;
    }
    else {
      if (initObj.hasOwnProperty('folder')) {
        this.folder = initObj.folder
      }
      else {
        this.folder = '';
      }
      if (initObj.hasOwnProperty('number_of_radiation_sources')) {
        this.number_of_radiation_sources = initObj.number_of_radiation_sources
      }
      else {
        this.number_of_radiation_sources = 0;
      }
      if (initObj.hasOwnProperty('number_of_rusting_models')) {
        this.number_of_rusting_models = initObj.number_of_rusting_models
      }
      else {
        this.number_of_rusting_models = 0;
      }
      if (initObj.hasOwnProperty('number_of_bending_models')) {
        this.number_of_bending_models = initObj.number_of_bending_models
      }
      else {
        this.number_of_bending_models = 0;
      }
      if (initObj.hasOwnProperty('max_bend_factor')) {
        this.max_bend_factor = initObj.max_bend_factor
      }
      else {
        this.max_bend_factor = 0.0;
      }
      if (initObj.hasOwnProperty('max_rust_factor')) {
        this.max_rust_factor = initObj.max_rust_factor
      }
      else {
        this.max_rust_factor = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GenRandomEnvironmentalEffectsRequest
    // Serialize message field [folder]
    bufferOffset = _serializer.string(obj.folder, buffer, bufferOffset);
    // Serialize message field [number_of_radiation_sources]
    bufferOffset = _serializer.int16(obj.number_of_radiation_sources, buffer, bufferOffset);
    // Serialize message field [number_of_rusting_models]
    bufferOffset = _serializer.int16(obj.number_of_rusting_models, buffer, bufferOffset);
    // Serialize message field [number_of_bending_models]
    bufferOffset = _serializer.int16(obj.number_of_bending_models, buffer, bufferOffset);
    // Serialize message field [max_bend_factor]
    bufferOffset = _serializer.float32(obj.max_bend_factor, buffer, bufferOffset);
    // Serialize message field [max_rust_factor]
    bufferOffset = _serializer.float32(obj.max_rust_factor, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GenRandomEnvironmentalEffectsRequest
    let len;
    let data = new GenRandomEnvironmentalEffectsRequest(null);
    // Deserialize message field [folder]
    data.folder = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [number_of_radiation_sources]
    data.number_of_radiation_sources = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [number_of_rusting_models]
    data.number_of_rusting_models = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [number_of_bending_models]
    data.number_of_bending_models = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [max_bend_factor]
    data.max_bend_factor = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [max_rust_factor]
    data.max_rust_factor = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.folder);
    return length + 18;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/GenRandomEnvironmentalEffectsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1601fbc011dbf3b713e5f6ae36624db7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string folder
    int16 number_of_radiation_sources
    int16 number_of_rusting_models
    int16 number_of_bending_models
    float32 max_bend_factor
    float32 max_rust_factor
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GenRandomEnvironmentalEffectsRequest(null);
    if (msg.folder !== undefined) {
      resolved.folder = msg.folder;
    }
    else {
      resolved.folder = ''
    }

    if (msg.number_of_radiation_sources !== undefined) {
      resolved.number_of_radiation_sources = msg.number_of_radiation_sources;
    }
    else {
      resolved.number_of_radiation_sources = 0
    }

    if (msg.number_of_rusting_models !== undefined) {
      resolved.number_of_rusting_models = msg.number_of_rusting_models;
    }
    else {
      resolved.number_of_rusting_models = 0
    }

    if (msg.number_of_bending_models !== undefined) {
      resolved.number_of_bending_models = msg.number_of_bending_models;
    }
    else {
      resolved.number_of_bending_models = 0
    }

    if (msg.max_bend_factor !== undefined) {
      resolved.max_bend_factor = msg.max_bend_factor;
    }
    else {
      resolved.max_bend_factor = 0.0
    }

    if (msg.max_rust_factor !== undefined) {
      resolved.max_rust_factor = msg.max_rust_factor;
    }
    else {
      resolved.max_rust_factor = 0.0
    }

    return resolved;
    }
};

class GenRandomEnvironmentalEffectsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GenRandomEnvironmentalEffectsResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GenRandomEnvironmentalEffectsResponse
    let len;
    let data = new GenRandomEnvironmentalEffectsResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/GenRandomEnvironmentalEffectsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GenRandomEnvironmentalEffectsResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: GenRandomEnvironmentalEffectsRequest,
  Response: GenRandomEnvironmentalEffectsResponse,
  md5sum() { return '1601fbc011dbf3b713e5f6ae36624db7'; },
  datatype() { return 'gazebo_radiation_plugins/GenRandomEnvironmentalEffects'; }
};
