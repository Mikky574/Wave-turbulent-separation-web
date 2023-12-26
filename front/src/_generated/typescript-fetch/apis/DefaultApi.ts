/* tslint:disable */
/* eslint-disable */
/**
 * FastAPI
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * The version of the OpenAPI document: 0.1.0
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


import * as runtime from '../runtime';
import type {
  HTTPValidationError,
  WsData,
} from '../models/index';
import {
    HTTPValidationErrorFromJSON,
    HTTPValidationErrorToJSON,
    WsDataFromJSON,
    WsDataToJSON,
} from '../models/index';

export interface UploadFileUploadPostRequest {
    u2: Blob;
    v2: Blob;
    w2: Blob;
    theta: Blob;
}

export interface WsPlaceholderWsPlaceholderPostRequest {
    wsData: WsData;
}

/**
 * 
 */
export class DefaultApi extends runtime.BaseAPI {

    /**
     * Upload File
     */
    async uploadFileUploadPostRaw(requestParameters: UploadFileUploadPostRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<{ [key: string]: string; }>> {
        if (requestParameters.u2 === null || requestParameters.u2 === undefined) {
            throw new runtime.RequiredError('u2','Required parameter requestParameters.u2 was null or undefined when calling uploadFileUploadPost.');
        }

        if (requestParameters.v2 === null || requestParameters.v2 === undefined) {
            throw new runtime.RequiredError('v2','Required parameter requestParameters.v2 was null or undefined when calling uploadFileUploadPost.');
        }

        if (requestParameters.w2 === null || requestParameters.w2 === undefined) {
            throw new runtime.RequiredError('w2','Required parameter requestParameters.w2 was null or undefined when calling uploadFileUploadPost.');
        }

        if (requestParameters.theta === null || requestParameters.theta === undefined) {
            throw new runtime.RequiredError('theta','Required parameter requestParameters.theta was null or undefined when calling uploadFileUploadPost.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        const consumes: runtime.Consume[] = [
            { contentType: 'multipart/form-data' },
        ];
        // @ts-ignore: canConsumeForm may be unused
        const canConsumeForm = runtime.canConsumeForm(consumes);

        let formParams: { append(param: string, value: any): any };
        let useForm = false;
        // use FormData to transmit files using content-type "multipart/form-data"
        useForm = canConsumeForm;
        // use FormData to transmit files using content-type "multipart/form-data"
        useForm = canConsumeForm;
        // use FormData to transmit files using content-type "multipart/form-data"
        useForm = canConsumeForm;
        // use FormData to transmit files using content-type "multipart/form-data"
        useForm = canConsumeForm;
        if (useForm) {
            formParams = new FormData();
        } else {
            formParams = new URLSearchParams();
        }

        if (requestParameters.u2 !== undefined) {
            formParams.append('u2', requestParameters.u2 as any);
        }

        if (requestParameters.v2 !== undefined) {
            formParams.append('v2', requestParameters.v2 as any);
        }

        if (requestParameters.w2 !== undefined) {
            formParams.append('w2', requestParameters.w2 as any);
        }

        if (requestParameters.theta !== undefined) {
            formParams.append('theta', requestParameters.theta as any);
        }

        const response = await this.request({
            path: `/upload`,
            method: 'POST',
            headers: headerParameters,
            query: queryParameters,
            body: formParams,
        }, initOverrides);

        return new runtime.JSONApiResponse<any>(response);
    }

    /**
     * Upload File
     */
    async uploadFileUploadPost(requestParameters: UploadFileUploadPostRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<{ [key: string]: string; }> {
        const response = await this.uploadFileUploadPostRaw(requestParameters, initOverrides);
        return await response.value();
    }

    /**
     * Ws Placeholder
     */
    async wsPlaceholderWsPlaceholderPostRaw(requestParameters: WsPlaceholderWsPlaceholderPostRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<any>> {
        if (requestParameters.wsData === null || requestParameters.wsData === undefined) {
            throw new runtime.RequiredError('wsData','Required parameter requestParameters.wsData was null or undefined when calling wsPlaceholderWsPlaceholderPost.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        const response = await this.request({
            path: `/__ws_placeholder`,
            method: 'POST',
            headers: headerParameters,
            query: queryParameters,
            body: WsDataToJSON(requestParameters.wsData),
        }, initOverrides);

        if (this.isJsonMime(response.headers.get('content-type'))) {
            return new runtime.JSONApiResponse<any>(response);
        } else {
            return new runtime.TextApiResponse(response) as any;
        }
    }

    /**
     * Ws Placeholder
     */
    async wsPlaceholderWsPlaceholderPost(requestParameters: WsPlaceholderWsPlaceholderPostRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<any> {
        const response = await this.wsPlaceholderWsPlaceholderPostRaw(requestParameters, initOverrides);
        return await response.value();
    }

}