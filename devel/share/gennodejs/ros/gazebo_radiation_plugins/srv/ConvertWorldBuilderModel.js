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

class ConvertWorldBuilderModelRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.world_builder_file = null;
      this.gazebo_model_path = null;
      this.noise_file = null;
    }
    else {
      if (initObj.hasOwnProperty('world_builder_file')) {
        this.world_builder_file = initObj.world_builder_file
      }
      else {
        this.world_builder_file = '';
      }
      if (initObj.hasOwnProperty('gazebo_model_path')) {
        this.gazebo_model_path = initObj.gazebo_model_path
      }
      else {
        this.gazebo_model_path = '';
      }
      if (initObj.hasOwnProperty('noise_file')) {
        this.noise_file = initObj.noise_file
      }
      else {
        this.noise_file = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ConvertWorldBuilderModelRequest
    // Serialize message field [world_builder_file]
    bufferOffset = _serializer.string(obj.world_builder_file, buffer, bufferOffset);
    // Serialize message field [gazebo_model_path]
    bufferOffset = _serializer.string(obj.gazebo_model_path, buffer, bufferOffset);
    // Serialize message field [noise_file]
    bufferOffset = _serializer.string(obj.noise_file, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ConvertWorldBuilderModelRequest
    let len;
    let data = new ConvertWorldBuilderModelRequest(null);
    // Deserialize message field [world_builder_file]
    data.world_builder_file = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [gazebo_model_path]
    data.gazebo_model_path = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [noise_file]
    data.noise_file = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.world_builder_file);
    length += _getByteLength(object.gazebo_model_path);
    length += _getByteLength(object.noise_file);
    return length + 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/ConvertWorldBuilderModelRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ca5346b14c3e8fe8c1488efbdb6aac5b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string world_builder_file
    string gazebo_model_path
    string noise_file
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ConvertWorldBuilderModelRequest(null);
    if (msg.world_builder_file !== undefined) {
      resolved.world_builder_file = msg.world_builder_file;
    }
    else {
      resolved.world_builder_file = ''
    }

    if (msg.gazebo_model_path !== undefined) {
      resolved.gazebo_model_path = msg.gazebo_model_path;
    }
    else {
      resolved.gazebo_model_path = ''
    }

    if (msg.noise_file !== undefined) {
      resolved.noise_file = msg.noise_file;
    }
    else {
      resolved.noise_file = ''
    }

    return resolved;
    }
};

class ConvertWorldBuilderModelResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ConvertWorldBuilderModelResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ConvertWorldBuilderModelResponse
    let len;
    let data = new ConvertWorldBuilderModelResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/ConvertWorldBuilderModelResponse';
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
    const resolved = new ConvertWorldBuilderModelResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: ConvertWorldBuilderModelRequest,
  Response: ConvertWorldBuilderModelResponse,
  md5sum() { return 'ca5346b14c3e8fe8c1488efbdb6aac5b'; },
  datatype() { return 'gazebo_radiation_plugins/ConvertWorldBuilderModel'; }
};
